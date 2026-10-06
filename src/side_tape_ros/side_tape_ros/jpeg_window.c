/*
 * Windowed JPEG decode for the side_tape perception node.
 *
 * Decodes only an MCU-aligned window (plus a safety margin) of a baseline YCbCr JPEG straight to BGR
 * using libjpeg-turbo's jpeg_crop_scanline / jpeg_skip_scanlines, stops after the last needed row,
 * then copies out exactly the requested w x h window.  Same libjpeg settings as OpenCV's decoder
 * (ISLOW IDCT, fancy upsampling), so the pixels are meant to equal cv2.imdecode(IMREAD_COLOR).
 *
 * Anything unusual (EXIF, progressive, non-YCbCr, any libjpeg warning or error, bad window) returns a
 * non-zero status and the caller falls back to cv2.imdecode.  libjpeg's error_exit longjmps; nothing
 * here can exit() or abort().
 *
 * Build: cc -O2 -shared -fPIC -o jpeg_window.so jpeg_window.c -ljpeg
 */
#include <setjmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

#define JW_OK 0
#define JW_BAD_ARGS 1
#define JW_UNSUPPORTED 2
#define JW_LIBJPEG_ERROR 3
#define JW_LIBJPEG_WARNING 4
#define JW_NOMEM 5

typedef struct {
    struct jpeg_error_mgr pub;
    jmp_buf jb;
    int warnings;
} jw_err;

static void jw_error_exit(j_common_ptr c) { longjmp(((jw_err *)c->err)->jb, 1); }
static void jw_output_message(j_common_ptr c) { (void)c; }
static void jw_emit_message(j_common_ptr c, int level) {
    if (level < 0) ((jw_err *)c->err)->warnings++;
}

static int imin(int a, int b) { return a < b ? a : b; }
static int imax(int a, int b) { return a > b ? a : b; }

/* Pure window math, exported so Python can unit-test its mirror.  r = {ax0, ay0, ax1, ay1}
 * (x1/y1 exclusive): the needed window grown by `margin` px, snapped outwards to the MCU grid,
 * clipped to the image.  Returns 0, or JW_BAD_ARGS if the window is empty or outside the image. */
int jpeg_window_plan(int W, int H, int mcu_w, int mcu_h, int x0, int y0, int w, int h, int margin, int *r) {
    if (W <= 0 || H <= 0 || mcu_w <= 0 || mcu_h <= 0 || margin < 0 || w <= 0 || h <= 0 || x0 < 0 || y0 < 0 ||
        x0 + w > W || y0 + h > H)
        return JW_BAD_ARGS;
    r[0] = imax(0, x0 - margin) / mcu_w * mcu_w;
    r[1] = imax(0, y0 - margin) / mcu_h * mcu_h;
    r[2] = imin(W, (x0 + w + margin + mcu_w - 1) / mcu_w * mcu_w);
    r[3] = imin(H, (y0 + h + margin + mcu_h - 1) / mcu_h * mcu_h);
    return JW_OK;
}

int jpeg_decode_window_m(const unsigned char *buf, unsigned long len, int x0, int y0, int w, int h,
                         unsigned char *out_bgr, int *img_w, int *img_h, int margin) {
    struct jpeg_decompress_struct cinfo;
    jw_err jerr;
    unsigned char *volatile row = NULL;
    volatile int created = 0;
    volatile int rc = JW_OK;

    if (!buf || !len || !out_bgr || w <= 0 || h <= 0 || x0 < 0 || y0 < 0 || margin < 0) return JW_BAD_ARGS;

    memset(&cinfo, 0, sizeof cinfo);
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jw_error_exit;
    jerr.pub.output_message = jw_output_message;
    jerr.pub.emit_message = jw_emit_message;
    jerr.warnings = 0;
    if (setjmp(jerr.jb)) {
        if (row) free(row);
        if (created) jpeg_destroy_decompress(&cinfo);
        return JW_LIBJPEG_ERROR;
    }
    jpeg_create_decompress(&cinfo);
    created = 1;
    jpeg_mem_src(&cinfo, buf, len);
    jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);
    jpeg_read_header(&cinfo, TRUE);

    if (img_w) *img_w = (int)cinfo.image_width;
    if (img_h) *img_h = (int)cinfo.image_height;
    for (jpeg_saved_marker_ptr m = cinfo.marker_list; m; m = m->next)
        if (m->marker == JPEG_APP0 + 1 && m->data_length >= 6 && memcmp(m->data, "Exif\0\0", 6) == 0) {
            rc = JW_UNSUPPORTED; /* OpenCV may auto-rotate on EXIF orientation */
            goto done;
        }
    if (cinfo.jpeg_color_space != JCS_YCbCr || cinfo.num_components != 3 || cinfo.progressive_mode ||
        cinfo.arith_code || cinfo.saw_Adobe_marker) {
        rc = JW_UNSUPPORTED;
        goto done;
    }
    int mcu_w = 8, mcu_h = 8;
    for (int i = 0; i < cinfo.num_components; i++) {
        mcu_w = imax(mcu_w, cinfo.comp_info[i].h_samp_factor * 8);
        mcu_h = imax(mcu_h, cinfo.comp_info[i].v_samp_factor * 8);
    }
    int r[4];
    if (jpeg_window_plan((int)cinfo.image_width, (int)cinfo.image_height, mcu_w, mcu_h, x0, y0, w, h, margin, r)) {
        rc = JW_BAD_ARGS;
        goto done;
    }

    cinfo.out_color_space = JCS_EXT_BGR;
    jpeg_start_decompress(&cinfo);
    JDIMENSION xo = (JDIMENSION)r[0], ww = (JDIMENSION)(r[2] - r[0]);
    jpeg_crop_scanline(&cinfo, &xo, &ww); /* already iMCU aligned; libjpeg may still widen it */
    if ((int)xo > x0 || (int)(xo + ww) < x0 + w) {
        rc = JW_LIBJPEG_ERROR;
        goto done;
    }
    row = (unsigned char *)malloc((size_t)ww * 3);
    if (!row) {
        rc = JW_NOMEM;
        goto done;
    }
    if (r[1] > 0 && jpeg_skip_scanlines(&cinfo, (JDIMENSION)r[1]) != (JDIMENSION)r[1]) {
        rc = JW_LIBJPEG_ERROR;
        goto done;
    }
    for (int y = r[1]; y < r[3]; y++) {
        JSAMPROW rp = row;
        if (jpeg_read_scanlines(&cinfo, &rp, 1) != 1) {
            rc = JW_LIBJPEG_ERROR;
            goto done;
        }
        if (y >= y0 && y < y0 + h) memcpy(out_bgr + (size_t)(y - y0) * w * 3, row + (size_t)(x0 - (int)xo) * 3, (size_t)w * 3);
    }
    if (jerr.warnings) rc = JW_LIBJPEG_WARNING;
done:
    if (row) free(row);
    jpeg_destroy_decompress(&cinfo); /* abort, not finish: we deliberately stop early */
    return rc;
}

int jpeg_decode_window(const unsigned char *buf, unsigned long len, int x0, int y0, int w, int h,
                       unsigned char *out_bgr, int *img_w, int *img_h) {
    return jpeg_decode_window_m(buf, len, x0, y0, w, h, out_bgr, img_w, img_h, 16);
}
