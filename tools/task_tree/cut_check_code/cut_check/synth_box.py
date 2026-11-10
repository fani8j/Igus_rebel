"""Draw a taped cardboard box and cut its seam by plan, with exact labels.

A scene is a top view of a box: two flaps meeting at a seam, clear packing
tape over the seam, optional stretch film, a shipping label, and a knife.
The seam is cut according to a plan of sections along it, from 0 % (where the
knife starts) to 100 % (where it ends):

    cut       the tape is cut through: an open dark slit       -> class 0 cut_ok
    partial   the blade only scratched it: a faint broken line  -> class 1 partial_cut
    miss      the blade missed: the tape is intact              -> class 2 not_cut

Because the cuts are drawn by the code, every pixel's true class is known,
so these images are exact training labels for YOLOv8 segmentation (class 3
is the knife). Each scene has random colours, lighting, tape, film, blur and
camera angle so the model learns the cut itself rather than one picture.

    from synth_box import Scene
    scene = Scene(seed=1)
    image, mask = scene.render([("cut", 0, 30), ("miss", 30, 45), ("cut", 45, 100)], knife_at=100)
"""
import cv2
import numpy as np

CLASS_ID = {"cut": 0, "partial": 1, "miss": 2}
KNIFE_ID = 3
BACKGROUND = 255
W, H = 960, 600
LINE_BAND = 0.035  # labelled cut-line width, fraction of image height (same as the real-video labels)


def parse_plan(text):
    """'cut:0-30,miss:30-45,cut:45-100' -> [('cut', 0, 30), ('miss', 30, 45), ('cut', 45, 100)]."""
    plan = []
    for part in text.split(","):
        state, rng = part.strip().split(":")
        a, b = (float(v) for v in rng.split("-"))
        if state not in CLASS_ID:
            raise ValueError(f"unknown state '{state}' (use cut, partial or miss)")
        plan.append((state, a, b))
    return sorted(plan, key=lambda p: p[1])


def random_plan(rng):
    """A random mix of cut / partial / missed sections covering 0-100 %."""
    cuts = sorted(rng.uniform(5, 95, size=rng.integers(1, 6)))
    edges = [0.0] + list(cuts) + [100.0]
    states = rng.choice(["cut", "cut", "partial", "miss", "miss"], size=len(edges) - 1)
    return [(s, a, b) for s, a, b in zip(states, edges[:-1], edges[1:]) if b - a > 1]


class Scene:
    """One box with fixed appearance; render() draws it at any cutting progress."""

    def __init__(self, seed=0, camera=True):
        self.rng = np.random.default_rng(seed)
        r = self.rng
        # box geometry (in the un-warped canvas)
        self.x0, self.x1 = int(r.uniform(60, 160)), int(r.uniform(800, 900))
        self.y0, self.y1 = int(r.uniform(50, 130)), int(r.uniform(470, 550))
        self.ys = int((self.y0 + self.y1) / 2 + r.uniform(-25, 25))  # seam
        self.tape_w = int(r.uniform(42, 72))
        self.reversed = bool(r.random() < 0.5)  # knife runs right-to-left
        self.seam_wobble = self._smooth_noise(self.x1 - self.x0 + 1, 1.2)
        self.base = self._draw_static()
        # camera: random view, brightness and colour
        self.H = self._random_homography() if camera else np.eye(3)
        self.rot90 = int(r.integers(0, 4)) if camera else 0
        self.gain = r.uniform(0.8, 1.2)
        self.bias = r.uniform(-18, 18)
        self.tint = 1 + r.uniform(-0.08, 0.08, size=3)
        self.blur = r.uniform(0, 1.3)
        self.noise = r.uniform(2, 6)

    # ---- helpers -------------------------------------------------------
    def _smooth_noise(self, n, amp):
        raw = self.rng.normal(0, 1, size=n // 20 + 3)
        return cv2.resize(raw.reshape(1, -1).astype(np.float32), (n, 1),
                          interpolation=cv2.INTER_CUBIC)[0] * amp

    def seam_point(self, pct):
        """Canvas (x, y) of the seam at pct % along the cut."""
        f = pct / 100.0
        if self.reversed:
            f = 1 - f
        x = self.x0 + f * (self.x1 - self.x0)
        y = self.ys + self.seam_wobble[int(np.clip(x - self.x0, 0, len(self.seam_wobble) - 1))]
        return x, y

    def to_image(self, x, y):
        """Where a canvas point ends up in the rendered (camera) image."""
        p = cv2.perspectiveTransform(np.float32([[[x, y]]]), self.H)[0, 0]
        x, y, w, h = float(p[0]), float(p[1]), W, H
        for _ in range(self.rot90):  # np.rot90 turns the picture 90 degrees counter-clockwise
            x, y, w, h = y, (w - 1) - x, h, w
        return x, y

    def seam_image_point(self, pct):
        """Pixel in the rendered image of the seam at pct % along the cut."""
        return self.to_image(*self.seam_point(pct))

    def _random_homography(self):
        r = self.rng
        src = np.float32([[0, 0], [W, 0], [W, H], [0, H]])
        jitter = r.uniform(-0.07, 0.07, size=(4, 2)) * [W, H]
        scale = r.uniform(0.85, 1.15)
        centre = np.float32([W / 2, H / 2])
        dst = (src - centre) * scale + centre + jitter
        ang = np.radians(r.uniform(-15, 15))
        rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]], np.float32)
        dst = (dst - centre) @ rot.T + centre
        return cv2.getPerspectiveTransform(src, dst.astype(np.float32))

    # ---- static appearance --------------------------------------------
    def _draw_static(self):
        r = self.rng
        img = np.zeros((H, W, 3), np.float32)
        # floor / table
        floor = np.array(r.choice([[120, 124, 128], [150, 150, 145], [95, 105, 115], [170, 165, 160]]), np.float32)
        img[:] = floor
        img += cv2.resize(r.normal(0, 10, (H // 40, W // 40)).astype(np.float32), (W, H))[..., None]
        img += r.normal(0, 3, (H, W, 1)).astype(np.float32)
        # cardboard
        card = np.array([r.uniform(70, 110), r.uniform(120, 160), r.uniform(170, 215)], np.float32)
        box = np.zeros((H, W, 3), np.float32) + card
        box += cv2.resize(r.normal(0, 9, (H // 30, W // 30)).astype(np.float32), (W, H))[..., None]
        box += r.normal(0, 4, (H, W, 1)).astype(np.float32)
        for _ in range(250):  # paper fibres
            x, y = int(r.uniform(0, W)), int(r.uniform(0, H))
            ln = int(r.uniform(5, 25))
            cv2.line(box, (x, y), (x + ln, y + int(r.uniform(-2, 2))), (card * 0.9).tolist(), 1)
        # the two flaps catch the light slightly differently
        box[:self.ys] *= r.uniform(0.95, 1.05)
        box[self.ys:] *= r.uniform(0.95, 1.05)
        inside = np.zeros((H, W), bool)
        inside[self.y0:self.y1, self.x0:self.x1] = True
        img[inside] = box[inside]
        # box outline shadow
        cv2.rectangle(img, (self.x0, self.y0), (self.x1, self.y1), (card * 0.55).tolist(), 2)
        # printed marks: shipping label with barcode, a circled number
        flap = r.choice([-1, 1])
        lx = int(r.uniform(self.x0 + 30, self.x0 + 0.45 * (self.x1 - self.x0)))
        ly = int(self.ys + flap * r.uniform(self.tape_w, self.tape_w + 40))
        lw, lh = int(r.uniform(160, 240)), int(r.uniform(70, 110))
        ly = ly if flap > 0 else ly - lh
        ly = int(np.clip(ly, self.y0 + 5, self.y1 - lh - 5))
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (238, 240, 240), -1)
        for k in range(int(lw * 0.6) // 3):
            if r.random() < 0.6:
                x = lx + 10 + 3 * k
                cv2.line(img, (x, ly + 10), (x, ly + lh - 28), (30, 30, 30), int(r.integers(1, 3)))
        cv2.putText(img, f"BOX{int(r.integers(100, 999))}", (lx + 10, ly + lh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 2)
        cx = int(r.uniform(self.x0 + 0.55 * (self.x1 - self.x0), self.x1 - 60))
        cy = int(self.ys - flap * r.uniform(self.tape_w + 20, self.tape_w + 60))
        cy = int(np.clip(cy, self.y0 + 40, self.y1 - 40))
        cv2.circle(img, (cx, cy), 30, (40, 40, 40), 2)
        cv2.putText(img, str(int(r.integers(10, 99))), (cx - 18, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (40, 40, 40), 2)
        # the seam (gap between the flaps)
        for x in range(self.x0, self.x1):
            y = self.ys + self.seam_wobble[x - self.x0]
            cv2.circle(img, (x, int(round(y))), 1, (card * 0.45).tolist(), -1)
        return img

    def _tape_and_film(self, img):
        r = np.random.default_rng(1234 + self.ys)  # same tape and film in every frame of a scene
        top, bottom = self.ys - self.tape_w // 2, self.ys + self.tape_w // 2
        band = np.zeros((H, W), np.float32)
        band[top:bottom, max(0, self.x0 - 25):min(W, self.x1 + 25)] = 1
        tint = np.array([150, 185, 205], np.float32)
        img = img * (1 - 0.18 * band[..., None]) + tint * 0.18 * band[..., None]
        glare = np.zeros((H, W), np.float32)
        for _ in range(int(r.uniform(15, 40))):  # shiny streaks along the tape
            x = int(r.uniform(self.x0 - 20, self.x1 + 20))
            y = int(r.uniform(top, bottom))
            cv2.ellipse(glare, (x, y), (int(r.uniform(15, 90)), int(r.uniform(1, 4))),
                        r.uniform(-12, 12), 0, 360, r.uniform(0.4, 1.0), -1)
        for y in (top, bottom):  # tape edges
            cv2.line(glare, (self.x0 - 25, y), (self.x1 + 25, y), 0.5, 1)
        glare = cv2.GaussianBlur(glare * band, (0, 0), 1.2)
        img = img + glare[..., None] * 120
        if r.random() < 0.6:  # stretch film: thin bright wrinkles over the whole box
            film = np.zeros((H, W), np.float32)
            for _ in range(int(r.uniform(20, 60))):
                pts = np.cumsum(r.normal(0, 18, (6, 2)), 0) + [r.uniform(0, W), r.uniform(0, H)]
                cv2.polylines(film, [pts.astype(np.int32)], False, r.uniform(0.2, 0.7), 1)
            img = img + cv2.GaussianBlur(film, (0, 0), 0.8)[..., None] * 90
        return img

    # ---- cuts and knife -------------------------------------------------
    def _cut_points(self, a, b):
        n = max(2, int(abs(b - a) * (self.x1 - self.x0) / 100 / 3))
        return np.array([self.seam_point(p) for p in np.linspace(a, b, n)], np.float32)

    def _draw_cut(self, img, state, a, b, r):
        pts = self._cut_points(a, b)
        if state == "cut":
            jag = pts + np.stack([np.zeros(len(pts)), r.normal(0, 0.6, len(pts))], 1)
            cv2.polylines(img, [jag.astype(np.int32)], False, (28, 32, 38), int(r.integers(3, 6)), cv2.LINE_AA)
            for off in (-3, 3):  # the two cut tape edges catch the light
                edge = jag + [0, off]
                cv2.polylines(img, [edge.astype(np.int32)], False, (215, 222, 226), 1, cv2.LINE_AA)
        elif state == "partial":
            i = 0
            while i < len(pts) - 1:
                ln = int(r.integers(2, 7))
                seg = pts[i:i + ln]
                if len(seg) >= 2:
                    cv2.polylines(img, [seg.astype(np.int32)], False, (58, 66, 74), int(r.integers(1, 3)),
                                  cv2.LINE_AA)
                i += ln + int(r.integers(2, 6))
        return img

    def _knife_polygon(self, pct):
        x, y = self.seam_point(pct)
        d = -1 if self.reversed else 1  # the holder trails behind the blade
        return np.array([[x, y], [x - d * 14, y - 10], [x - d * 40, y - 60], [x - d * 40, y - 150],
                         [x - d * 95, y - 150], [x - d * 95, y - 55], [x - d * 30, y + 6]], np.float32)

    def render(self, plan, knife_at=None, progress=100.0):
        """Image and class mask with the seam cut up to `progress` % according to `plan`.

        knife_at: % along the seam where the knife is drawn, or None for no knife.
        Seam parts beyond `progress` are not cut yet (labelled not_cut, like a miss).
        """
        r = np.random.default_rng(99)
        img = self._tape_and_film(self.base.copy())
        mask = np.full((H, W), BACKGROUND, np.uint8)
        band = max(4, int(LINE_BAND * H))
        for state, a, b in plan:
            done_b = min(b, progress)
            if done_b > a:
                img = self._draw_cut(img, state, a, done_b, np.random.default_rng(int(a * 100)))
                cv2.polylines(mask, [self._cut_points(a, done_b).astype(np.int32)], False, CLASS_ID[state], band)
            if b > progress:
                cv2.polylines(mask, [self._cut_points(max(a, progress), b).astype(np.int32)], False,
                              CLASS_ID["miss"], band)
        if knife_at is not None:
            poly = self._knife_polygon(knife_at)
            cv2.fillPoly(img, [poly.astype(np.int32)], (48, 50, 56))
            cv2.polylines(img, [poly.astype(np.int32)], True, (90, 92, 98), 2)
            tip = poly[0]
            blade = np.array([tip, poly[1], [tip[0] - (-1 if self.reversed else 1) * 4, tip[1] - 16]], np.int32)
            cv2.fillPoly(img, [blade], (205, 210, 215))
            cv2.fillPoly(mask, [poly.astype(np.int32)], KNIFE_ID)
        # camera and lighting
        img = img * self.gain + self.bias
        img = img * self.tint
        if self.blur > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), self.blur)
        img = img + r.normal(0, self.noise, img.shape)
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cv2.warpPerspective(img, self.H, (W, H), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpPerspective(mask, self.H, (W, H), flags=cv2.INTER_NEAREST, borderValue=BACKGROUND)
        if self.rot90:
            img, mask = np.rot90(img, self.rot90).copy(), np.rot90(mask, self.rot90).copy()
        return img, mask


# ---- the real box from the overhead scene camera ------------------------------
# Annotated by hand on videos/scene.mp4 (960x540): the top face of the carton and
# the seam line through the clear tape. Only frames 0-29 show the box with no arm
# or person in front of it; frames 0-21 are used for training, 24-29 for testing.
SCENE_VIDEO = "videos/scene.mp4"
SCENE_BOX = [(385, 336), (582, 341), (590, 476), (355, 470)]
SCENE_SEAM = [(373, 400), (480, 401), (585, 402)]
# The two box edges next to the top seam (the short sides of the top face, where
# the tape folds over the corners), from the back corner to the front corner.
# Each passes through one end of the seam.
SCENE_LEFT_EDGE = [(388, 338), (372, 400), (352, 466)]
SCENE_RIGHT_EDGE = [(580, 342), (585, 402), (590, 472)]
SCENE_LINES = {"top": SCENE_SEAM, "left": SCENE_LEFT_EDGE, "right": SCENE_RIGHT_EDGE}
SCENE_TRAIN_FRAMES = range(0, 22)
SCENE_TEST_FRAMES = range(24, 30)
SCENE_CROP = (330, 310, 630, 540)  # x0, y0, x1, y1 around the box
SCENE_ZOOM = 3.0


class RealBoxScene(Scene):
    """The real carton from the scene camera, with cuts drawn along its real seam."""

    def __init__(self, frame_index, seed=0, vary=True, video=SCENE_VIDEO):
        self.rng = r = np.random.default_rng(seed)
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise SystemExit(f"cannot read frame {frame_index} of {video}")
        x0, y0, x1, y1 = SCENE_CROP
        if vary:  # move and resize the crop a little so the box is not always in the same place
            dx, dy = r.uniform(-25, 25, size=2)
            x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
        self.crop = (int(x0), int(y0), int(x1), int(y1))
        self.zoom = SCENE_ZOOM * (r.uniform(0.85, 1.15) if vary else 1.0)
        cx0, cy0, cx1, cy1 = self.crop
        self.base = cv2.resize(frame[cy0:cy1, cx0:cx1], None, fx=self.zoom, fy=self.zoom,
                               interpolation=cv2.INTER_CUBIC).astype(np.float32)
        self.lines = {}
        for name, pts in SCENE_LINES.items():
            px = np.float32([self.to_image(x, y) for x, y in pts])
            seg = np.linalg.norm(np.diff(px, axis=0), axis=1)
            self.lines[name] = (px, np.concatenate([[0], np.cumsum(seg)]) / seg.sum(), int(seg.sum()))
        self.directions = {name: bool(r.random() < 0.5) if vary else False for name in SCENE_LINES}
        self.use_line("top")
        self.gain = r.uniform(0.85, 1.15) if vary else 1.0
        self.bias = r.uniform(-12, 12) if vary else 0.0
        self.tint = 1 + (r.uniform(-0.05, 0.05, size=3) if vary else np.zeros(3))
        self.blur = r.uniform(0.6, 1.4) if vary else 0.9
        self.noise = r.uniform(1.5, 4) if vary else 2.0

    def to_image(self, x, y):
        """Scene-video pixel -> pixel in the cropped, enlarged image."""
        return (x - self.crop[0]) * self.zoom, (y - self.crop[1]) * self.zoom

    def seam_point(self, pct):
        f = pct / 100.0
        if self.reversed:
            f = 1 - f
        k = int(np.clip(np.searchsorted(self.seam_cum, f) - 1, 0, len(self.seam_px) - 2))
        t = (f - self.seam_cum[k]) / max(1e-6, self.seam_cum[k + 1] - self.seam_cum[k])
        x, y = self.seam_px[k] + t * (self.seam_px[k + 1] - self.seam_px[k])
        return float(x), float(y)

    def use_line(self, name):
        """Make seam_point() and the cut drawing work on one of SCENE_LINES."""
        self.line = name
        self.seam_px, self.seam_cum, length = self.lines[name]
        self.x0, self.x1 = 0, length  # line length, used for point spacing
        self.reversed = self.directions[name]

    def seam_image_point(self, pct, line=None):
        if line is not None:
            self.use_line(line)
        return self.seam_point(pct)  # this scene draws directly in image pixels

    def _knife_polygon(self, pct):
        x, y = self.seam_point(pct)
        if self.line == "top":
            return super()._knife_polygon(pct) * [1, 0.8] + [0, 0.2 * y]
        d = -1 if self.line == "left" else 1  # the holder reaches in from outside the box
        return np.array([[x, y], [x + d * 12, y - 12], [x + d * 55, y - 40], [x + d * 150, y - 40],
                         [x + d * 150, y + 18], [x + d * 50, y + 18], [x + d * 8, y + 8]], np.float32)

    def render(self, plan, knife_at=None, progress=100.0):
        """Cut only the top seam (the edges are left untouched and unlabelled)."""
        return self.render_lines({"top": plan}, {"top": progress}, ("top", knife_at) if knife_at is not None
                                 else None)

    def render_lines(self, plans, progress, knife=None):
        """Cut several lines. plans/progress: {line: plan / % done}; knife: (line, %) or None.

        Lines in `plans` are labelled along their whole length: done parts by
        their plan state, parts not reached yet as not cut.
        """
        r = np.random.default_rng(99)
        img = self.base.copy()
        h, w = img.shape[:2]
        mask = np.full((h, w), BACKGROUND, np.uint8)
        band = max(4, int(LINE_BAND * h))
        for line, plan in plans.items():
            self.use_line(line)
            done = progress.get(line, 100.0)
            for state, a, b in plan:
                done_b = min(b, done)
                if done_b > a:
                    img = self._draw_cut(img, state, a, done_b, np.random.default_rng(int(a * 100) + len(line)))
                    cv2.polylines(mask, [self._cut_points(a, done_b).astype(np.int32)], False, CLASS_ID[state],
                                  band)
                if b > done:
                    cv2.polylines(mask, [self._cut_points(max(a, done), b).astype(np.int32)], False,
                                  CLASS_ID["miss"], band)
        if knife is not None:
            self.use_line(knife[0])
            poly = self._knife_polygon(knife[1])
            cv2.fillPoly(img, [poly.astype(np.int32)], (48, 50, 56))
            cv2.polylines(img, [poly.astype(np.int32)], True, (90, 92, 98), 2)
            cv2.fillPoly(mask, [poly.astype(np.int32)], KNIFE_ID)
        self.use_line("top")
        img = cv2.GaussianBlur(img, (0, 0), self.blur)  # blend the drawn cut into the soft camera image
        img = (img * self.gain + self.bias) * self.tint + r.normal(0, self.noise, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8), mask
