#!/usr/bin/env python3
# Provenance: moved into box_seam_detection (2026-10-12) from seam_handoff_20260908/scripts/query_depth_at_pixel.py
# at commit 08d1041 (working tree identical to HEAD, unmodified).
# Changes: the legacy --image/--roi/--auto-roi/--detect-debug/--detection-output-dir mode (detect_tape_seam.py) was dropped.
"""query_depth_at_pixel.py

從 RGB-D 點雲 PCD 檔案中，依照 RGB 影像上的像素座標 (u, v)，
查出該像素對應的 3D 座標（相機座標系下的 X, Y, Z，單位：公尺）。

用途
----
detect_tape_seam.py 算出的 seam_line（left_mid, right_mid）只是 2D 像素座標，
必須先靠這支程式接上「同一次拍攝」存下來的點雲，補上深度資訊，才能進一步
透過 TF2 轉換到機械手臂座標系（詳見 CameraExtrinsicsSetup.md）。

支援兩種點雲，自動判斷：
----------------------
1. Organized（PCD header 的 HEIGHT > 1，點的排列順序跟某張影像的掃描順序
   一一對應）：直接用 (u, v) 索引查表，最快最準。

2. Unorganized（HEIGHT = 1，例如 Orbbec 的「彩色點雲」/camera/depth_registered/points
   —— 這個 topic 實際上是驅動內部把深度視角/彩色視角沒有重疊、或找不到對應
   顏色的點濾掉之後，輸出成一個攤平的清單，不是一張跟著影像排列的網格。
   這種情況沒有 (u, v) 可以直接索引，改用「反投影搜尋」：把點雲裡每個有效
   點的 (X, Y, Z) 用彩色相機內參投影回彩色影像平面，找出投影後最接近目標
   像素的那個點）。這個模式需要額外提供彩色相機內參 --fx --fy --cx --cy，
   可用以下指令查詢（輸出的 k 矩陣依序對應 fx, 0, cx, 0, fy, cy, 0, 0, 1）：
       ros2 topic echo /camera/color/camera_info --once

三種取得目標像素的方式，可任選一種（--pixel 可以跟另外兩種之一合併使用）：
----------------------------------------------------------------
1. --pixel U V：直接指定單一像素座標。
2. --seam-json：讀 detect_tape_seam.py 事先存好的 <base>_tape_seam.json，
   自動取裡面的 seam_line（左中點、右中點）兩個座標。
3. --image：直接給一張 RGB 影像，程式會在記憶體中呼叫 detect_tape_seam.py
   的偵測邏輯（跟直接執行 detect_tape_seam.py 完全相同的演算法），算出
   seam_line 後立刻接著查深度——不需要事先手動跑一次 detect_tape_seam.py
   或準備 JSON 檔。這個模式一樣會存出 <base>_tape_seam.png / .json（跟
   detect_tape_seam.py 的輸出一致），方便事後檢查偵測是否正確，只是這裡
   會多做「查深度」這一步。detect_tape_seam.py 必須跟本檔案放在同一個
   資料夾（用來 import 裡面的偵測函式）。--seam-json 與 --image 不能同時
   使用（兩者都是取得 seam_line 的方式，只需要選一種）。

用法
----
    # Organized 點雲：查詢單一像素
    python3 query_depth_at_pixel.py --pcd frame.pcd --pixel 649 159

    # Unorganized 點雲（例如 depth_registered/points）：需要加上相機內參
    python3 query_depth_at_pixel.py --pcd frame.pcd --pixel 649 159 \\
        --fx 900.5 --fy 900.3 --cx 640.1 --cy 400.7

    # 讀 detect_tape_seam.py 事先存好的 JSON，一次查 left_mid / right_mid 兩點
    python3 query_depth_at_pixel.py --pcd frame.pcd --seam-json frame_tape_seam.json \\
        --fx 900.5 --fy 900.3 --cx 640.1 --cy 400.7

    # 直接給圖片：偵測 + 查深度一次完成，不用先手動跑 detect_tape_seam.py
    python3 query_depth_at_pixel.py --pcd frame.pcd --image frame.png \\
        --fx 900.5 --fy 900.3 --cx 640.1 --cy 400.7

    # --image 也可以指定 ROI（不指定則沿用 detect_tape_seam.py 的 DEFAULT_ROI）
    python3 query_depth_at_pixel.py --pcd frame.pcd --image frame.png --roi 600 100 870 310 \\
        --fx 900.5 --fy 900.3 --cx 640.1 --cy 400.7

    # 合併使用並把結果存檔
    python3 query_depth_at_pixel.py --pcd frame.pcd \\
        --image frame.png --pixel 700 150 \\
        --fx 900.5 --fy 900.3 --cx 640.1 --cy 400.7 \\
        --output-json frame_seam_3d.json

備註
----
- Organized 模式下，若目標像素本身深度無效（NaN），會自動往外以同心方框
  搜尋最近的有效深度點代替（預設最大半徑 15 像素），並在結果裡註明。
- Unorganized（反投影）模式下，若最近的點反投影後跟目標像素距離超過
  --max-pixel-distance（預設 3 像素），視為找不到有效對應，避免誤用一個
  離目標很遠的點。
- 若查到的點深度有效、但顏色是黑色 (0, 0, 0)，代表這個像素落在深度視角與
  彩色視角沒有重疊的區域，顏色是相機驅動補上的預設值，不是真的黑色物體，
  x/y/z 仍然可以正常使用（僅 organized 模式會遇到這種情況；unorganized
  模式下這類點本來就已經被驅動濾掉，不會出現在點雲裡）。
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np


class PCDReadError(RuntimeError):
    """PCD 檔案格式不符或內容不完整時拋出。"""


_TYPE_MAP = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
}


def _parse_pcd_header(f):
    """讀取 PCD v0.7 文字 header，回傳 {欄位名: [值...]} 的字典。"""
    header = {}
    while True:
        raw_line = f.readline()
        if not raw_line:
            raise PCDReadError("檔案在讀完 header 前就結束了，PCD 檔可能不完整")
        line = raw_line.decode("ascii", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        header[key] = rest.split()
        if key == "DATA":
            break

    required = ["FIELDS", "SIZE", "TYPE", "COUNT", "WIDTH", "HEIGHT", "POINTS", "DATA"]
    missing = [key for key in required if key not in header]
    if missing:
        raise PCDReadError(f"PCD header 缺少必要欄位: {', '.join(missing)}")

    return header


def _build_dtype(header):
    """依 FIELDS/SIZE/TYPE/COUNT 組出對應的 numpy structured dtype。"""
    fields = header["FIELDS"]
    sizes = [int(s) for s in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(c) for c in header["COUNT"]]

    dtype_fields = []
    for name, size, type_char, count in zip(fields, sizes, types, counts):
        np_type = _TYPE_MAP.get((type_char, size))
        if np_type is None:
            raise PCDReadError(f"不支援的欄位型別: {name} ({type_char}{size * 8})")
        if count == 1:
            dtype_fields.append((name, np_type))
        else:
            dtype_fields.append((name, np_type, (count,)))
    return np.dtype(dtype_fields)


def load_pcd(pcd_path):
    """讀取 PCD 檔，回傳 (header_dict, points_flat, width, height)。

    points_flat 是長度 WIDTH*HEIGHT 的一維 structured array，保留檔案原本
    的點序。呼叫端依 height 是否 > 1 決定要用 organized 索引還是反投影搜尋。
    """
    pcd_path = Path(pcd_path)
    with open(pcd_path, "rb") as f:
        header = _parse_pcd_header(f)
        data_kind = header["DATA"][0]
        width = int(header["WIDTH"][0])
        height = int(header["HEIGHT"][0])
        n_points = int(header["POINTS"][0])
        dtype = _build_dtype(header)

        if width * height != n_points:
            raise PCDReadError(
                f"WIDTH*HEIGHT ({width}x{height}={width * height}) 與 POINTS "
                f"({n_points}) 對不上，檔案可能損毀"
            )

        if data_kind == "binary":
            raw = f.read(n_points * dtype.itemsize)
            if len(raw) != n_points * dtype.itemsize:
                raise PCDReadError("binary 資料長度與 header 宣告的點數不一致，檔案可能不完整")
            points = np.frombuffer(raw, dtype=dtype)
        elif data_kind == "ascii":
            points = np.zeros(n_points, dtype=dtype)
            field_names = header["FIELDS"]
            for i in range(n_points):
                line = f.readline().decode("ascii").strip()
                if not line:
                    raise PCDReadError(f"ASCII 資料在第 {i} 個點就結束了，檔案可能不完整")
                values = line.split()
                col = 0
                for name in field_names:
                    sub_shape = dtype[name].shape
                    count = sub_shape[0] if sub_shape else 1
                    if count == 1:
                        points[i][name] = float(values[col])
                        col += 1
                    else:
                        points[i][name] = [float(v) for v in values[col:col + count]]
                        col += count
        else:
            raise PCDReadError(f"不支援的 DATA 格式: {data_kind}（僅支援 binary / ascii）")

    return header, points, width, height


def unpack_rgb(value):
    """把 PCL 慣例的 packed rgb/rgba 欄位還原成 (r, g, b)，範圍 0-255。"""
    arr = np.asarray(value)
    if arr.dtype == np.float32:
        packed = int(arr.view(np.uint32).item())
    else:
        packed = int(arr.item())
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    return r, g, b


def _color_field_name(points):
    if "rgb" in points.dtype.names:
        return "rgb"
    if "rgba" in points.dtype.names:
        return "rgba"
    return None


# ---------------------------------------------------------------------------
# 模式一：organized 點雲，直接用 (u, v) 索引
# ---------------------------------------------------------------------------

def query_pixel(points_2d, u, v):
    """查詢單一像素 (u, v) 對應的 3D 座標與顏色（若有）。"""
    height, width = points_2d.shape
    if not (0 <= u < width) or not (0 <= v < height):
        raise ValueError(
            f"像素座標 ({u}, {v}) 超出點雲範圍（WIDTH={width}, HEIGHT={height}）"
        )

    point = points_2d[v, u]
    x, y, z = float(point["x"]), float(point["y"]), float(point["z"])

    result = {"u": u, "v": v, "x": x, "y": y, "z": z}
    # 2026-09-02 修正：這台相機的 organized 點雲，部分「沒對到深度」的像素
    # 存的不是 NaN，而是塞 (0, 0, 0) 當佔位值（常見於箱子輪廓邊緣這種深度
    # 感測器容易失效的地方）。只檢查 isnan() 會把這種佔位值誤判成「有效」，
    # 導致 query_pixel_with_fallback() 不會去附近找真正有效的點，讓 (0,0,0)
    # 一路傳到 base_link 座標系（camera_point_to_base_link.py 對 (0,0,0) 做
    is_nonfinite = not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z))
    is_zero_sentinel = (x == 0.0 and y == 0.0 and z == 0.0)
    result["valid"] = not (is_nonfinite or z <= 0.0 or is_zero_sentinel)
    if is_zero_sentinel and not is_nonfinite:
        result["note"] = (
            "深度為 (0,0,0)，判定為感測器沒對到深度的佔位值（常見於物體輪廓"
            "邊緣），視為無效，會嘗試往外搜尋最近的有效深度點代替"
        )

    color_field = _color_field_name(points_2d)
    if color_field is not None:
        r, g, b = unpack_rgb(point[color_field])
        result["rgb"] = (r, g, b)
        if result["valid"] and (r, g, b) == (0, 0, 0):
            result["note"] = (
                "深度有效，但顏色是黑色 (0,0,0)——這個像素落在深度視角與彩色"
                "視角沒有重疊的區域，顏色欄位是相機驅動補上的預設黑色，不是"
                "真的黑色物體，x/y/z 仍可放心使用"
            )

    return result


# 2026-09-02 新增：箱子四個角點兩兩成對構成 left_edge/right_edge 這兩條「綠
# 線」（見 detect_tape_seam.py 輸出），對切割任務來說，back_left/back_right
# 這種角點只是「這條邊的起點」，不要求非得是幾何上最精確的那個角——沿著同一
# 條邊線移動的點物理意義相同（見 vision_pick_execute.py 裡 seg1/seg3 沿
# left_edge/right_edge 切過去的邏輯）。角點正好是箱子跟背景的輪廓交界，是
# 深度感測器最容易失效的地方；反過來說，沿著邊線往箱子內部（另一端角點的方
# 向）走一小段，幾乎保證還踩在箱子表面上，不會像「往任意方向的方框搜尋」那
# 樣有機會找到背景或箱子上方雜物的有效深度、卻其實已經不在箱子上了。
_EDGE_PARTNER = {
    "back_left": "front_left",
    "back_right": "front_right",
    "front_left": "back_left",
    "front_right": "back_right",
}

# 沿邊線內縮最多不超過邊線長度的 15%，避免內縮太多導致下刀起點跟真實箱角
# 差太多；超過這個上限還找不到有效深度，才退回原本「往任意方向的方框搜尋」
# 當最後手段（保留既有 fallback 當安全網，不是被取代）。
_LINE_FALLBACK_MAX_FRACTION = 0.15


def _line_step_points(u0, v0, u1, v1, max_fraction, step_px=2.0):
    """從 (u0, v0) 沿直線往 (u1, v1) 方向走，每次前進 step_px 像素，直到
    max_fraction * 兩點總距離為止，回傳依序嘗試的整數像素座標列表（由近到
    遠，不含起點本身）。兩點重合（長度接近零）時回傳空列表。"""
    dx, dy = u1 - u0, v1 - v0
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return []
    max_dist = length * max_fraction
    points = []
    dist = step_px
    while dist <= max_dist + 1e-6:
        frac = dist / length
        uu, vv = int(round(u0 + dx * frac)), int(round(v0 + dy * frac))
        if not points or points[-1] != (uu, vv):
            points.append((uu, vv))
        dist += step_px
    return points


def _nearest_valid_pixel(points_2d, u, v, max_radius):
    """在 (u, v) 深度無效時，往外以同心方框搜尋最近的有效深度點座標。
    跟 query_pixel() 用同一套「有效」判定（2026-09-02 更新）：NaN 或
    (0,0,0) 佔位值都算無效，避免搜尋圈到另一個同樣沒查到深度的佔位點，
    誤判成「找到了」。"""
    height, width = points_2d.shape
    for radius in range(1, max_radius + 1):
        u_min, u_max = max(0, u - radius), min(width - 1, u + radius)
        v_min, v_max = max(0, v - radius), min(height - 1, v + radius)
        for vv in range(v_min, v_max + 1):
            for uu in range(u_min, u_max + 1):
                # 只掃這一圈的邊框，避免重複掃到已經檢查過的內圈
                if abs(uu - u) != radius and abs(vv - v) != radius:
                    continue
                point = points_2d[vv, uu]
                x, y, z = float(point["x"]), float(point["y"]), float(point["z"])
                is_nonfinite = not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z))
                is_zero_sentinel = (x == 0.0 and y == 0.0 and z == 0.0)
                if not (is_nonfinite or z <= 0.0 or is_zero_sentinel):
                    return uu, vv
    return None


def query_pixel_with_fallback(points_2d, u, v, max_radius=15, edge_partner_uv=None,
                               line_max_fraction=_LINE_FALLBACK_MAX_FRACTION):
    """查詢像素 (u, v)；若深度無效，自動改用附近最近的有效深度點。

    edge_partner_uv：若這個點是箱子角點、且知道它所在那條邊線的另一端角點
    像素座標，優先沿著邊線往那個方向內縮（最多 line_max_fraction 比例）找
    有效深度點——這個方向保證還在箱子表面上（見上方 `_EDGE_PARTNER` 的說
    明）。沿線內縮找不到才退回原本的方框搜尋（`_nearest_valid_pixel`）當最
    後手段，不影響沒有 edge_partner_uv 的呼叫端（例如 left_mid/right_mid、
    單一 --pixel 查詢）的既有行為。
    """
    result = query_pixel(points_2d, u, v)
    if result["valid"]:
        return result

    if edge_partner_uv is not None:
        height, width = points_2d.shape
        ou, ov = edge_partner_uv
        edge_length = math.hypot(ou - u, ov - v)
        for uu, vv in _line_step_points(u, v, ou, ov, line_max_fraction):
            if not (0 <= uu < width) or not (0 <= vv < height):
                continue
            candidate = query_pixel(points_2d, uu, vv)
            if candidate["valid"]:
                distance = math.hypot(uu - u, vv - v)
                pct = (distance / edge_length * 100.0) if edge_length > 1e-6 else 0.0
                candidate["note"] = (
                    f"原始角點 ({u}, {v}) 深度無效（常見於箱子輪廓邊緣），改沿邊線"
                    f"往另一端內縮 {distance:.1f} 像素（邊線長度的 {pct:.1f}%，"
                    f"上限 {line_max_fraction * 100:.0f}%）找到有效深度點代替——"
                    "這個方向保證還在箱子表面上，比方框搜尋更可靠"
                )
                candidate["requested_pixel"] = {"u": u, "v": v}
                return candidate
        # 沿線內縮到上限都沒找到有效深度，繼續往下走原本的方框搜尋。

    nearest = _nearest_valid_pixel(points_2d, u, v, max_radius=max_radius)
    if nearest is None:
        result["note"] = (
            f"({u}, {v}) 本身及其附近 {max_radius} 像素內都沒有有效深度，"
            "建議換個角度、調整相機距離，或避開反光/背光表面後重拍"
        )
        return result

    nu, nv = nearest
    fallback = query_pixel(points_2d, nu, nv)
    distance = max(abs(nu - u), abs(nv - v))
    replacement_note = (
        f"原始像素 ({u}, {v}) 深度無效（可能是邊緣的 flying pixel 或反光造成），"
        f"改用 {distance} 像素外的最近有效點 ({nu}, {nv}) 代替"
    )
    existing_note = fallback.get("note")
    fallback["note"] = (
        replacement_note if not existing_note else f"{replacement_note}；另外，{existing_note}"
    )
    fallback["requested_pixel"] = {"u": u, "v": v}
    return fallback


# ---------------------------------------------------------------------------
# 模式二：unorganized 點雲，反投影到彩色影像平面後找最近點
# ---------------------------------------------------------------------------

def query_by_reprojection(points_flat, u0, v0, fx, fy, cx, cy, max_pixel_distance=3.0):
    """把點雲裡每個有效點投影回彩色影像平面，找出離 (u0, v0) 最近的點。

    假設點雲座標系跟彩色相機的光學座標系一致（Orbbec 的
    /camera/depth_registered/points 依官方文件為「registered to the color
    image frame」，符合此假設）。
    """
    x = points_flat["x"].astype(np.float64)
    y = points_flat["y"].astype(np.float64)
    z = points_flat["z"].astype(np.float64)

    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return {
            "u": u0, "v": v0, "valid": False,
            "note": "點雲裡沒有任何 z > 0 的有效點，無法反投影搜尋",
        }

    idx_valid = np.nonzero(valid)[0]
    zz = z[idx_valid]
    proj_u = fx * x[idx_valid] / zz + cx
    proj_v = fy * y[idx_valid] / zz + cy

    dist2 = (proj_u - u0) ** 2 + (proj_v - v0) ** 2
    best_local = int(np.argmin(dist2))
    best_idx = idx_valid[best_local]
    best_dist = float(np.sqrt(dist2[best_local]))

    result = {
        "u": u0,
        "v": v0,
        "reprojected_u": float(proj_u[best_local]),
        "reprojected_v": float(proj_v[best_local]),
        "pixel_distance": best_dist,
        "x": float(x[best_idx]),
        "y": float(y[best_idx]),
        "z": float(z[best_idx]),
    }

    color_field = _color_field_name(points_flat)
    if color_field is not None:
        r, g, b = unpack_rgb(points_flat[color_field][best_idx])
        result["rgb"] = (r, g, b)

    if best_dist > max_pixel_distance:
        result["valid"] = False
        result["note"] = (
            f"最近的點反投影後距離目標像素 {best_dist:.1f} 像素，超過容許值 "
            f"{max_pixel_distance} 像素，這個位置附近可能本來就沒有有效的彩色點"
            "（例如落在深度/彩色視角沒有重疊的區域），建議換個角度或距離重拍"
        )
    else:
        result["valid"] = True
        result["note"] = (
            f"反投影最近點，跟目標像素相差 {best_dist:.2f} 像素"
        )

    return result


def load_capture_manifest(manifest_path, pcd_path):
    try:
        with open(manifest_path, encoding="utf-8") as input_file:
            manifest = json.load(input_file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read capture manifest {manifest_path}: {error}") from error
    if manifest.get("schema") != "rgbd_capture/v1":
        raise ValueError(f"unsupported capture manifest schema: {manifest.get('schema')!r}")
    cloud = manifest.get("pointcloud", {})
    image = manifest.get("image", {})
    camera_info = manifest.get("camera_info", {})
    if Path(cloud.get("path", "")).resolve() != Path(pcd_path).resolve():
        raise ValueError("PCD does not belong to capture manifest")
    image_frame = image.get("header", {}).get("frame_id")
    if not image_frame or image_frame != cloud.get("header", {}).get("frame_id") or \
            image_frame != camera_info.get("header", {}).get("frame_id"):
        raise ValueError("capture manifest RGB, PointCloud2 and CameraInfo frames differ")
    if image.get("width") != camera_info.get("width") or image.get("height") != camera_info.get("height"):
        raise ValueError("capture manifest CameraInfo dimensions differ from RGB dimensions")
    if not isinstance(cloud.get("header", {}).get("stamp_ns"), int):
        raise ValueError("capture manifest is missing PointCloud2 acquisition timestamp")
    return manifest


def _camera_intrinsics(camera_info):
    projection = np.asarray(camera_info.get("p", ()), dtype=np.float64)
    if projection.size == 12 and projection[0] > 0.0 and projection[5] > 0.0:
        return float(projection[0]), float(projection[5]), float(projection[2]), float(projection[6])
    matrix = np.asarray(camera_info.get("k", ()), dtype=np.float64)
    if matrix.size == 9 and matrix[0] > 0.0 and matrix[4] > 0.0:
        return float(matrix[0]), float(matrix[4]), float(matrix[2]), float(matrix[5])
    raise ValueError("capture manifest CameraInfo lacks valid projection intrinsics")


def resolve_intrinsics(manifest, args):
    cli = (args.fx, args.fy, args.cx, args.cy)
    supplied = [value is not None for value in cli]
    if any(supplied) and not all(supplied):
        raise ValueError("--fx --fy --cx --cy must be supplied together")
    if manifest is not None:
        resolved = _camera_intrinsics(manifest["camera_info"])
        if all(supplied) and not np.allclose(cli, resolved, rtol=0.0, atol=1e-6):
            raise ValueError("CLI intrinsics do not match capture manifest CameraInfo")
    elif all(supplied):
        resolved = tuple(float(value) for value in cli)
    else:
        raise ValueError("unorganized point cloud requires capture manifest or --fx --fy --cx --cy")
    fx, fy, cx, cy = resolved
    if not np.all(np.isfinite(resolved)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera intrinsics must be finite with positive fx and fy")
    return resolved


# ---------------------------------------------------------------------------
# Lens distortion (2026-09-28)
# ---------------------------------------------------------------------------
# The registered RGB image is the RAW (distorted) color image, so detector
# pixels are distorted pixels. Named pixels are undistorted before casting
# rays, and cloud points are projected WITH distortion, so the eroded-polygon
# interior selection and local-support counts live in the same image geometry
# as the detector pixels. The forward/inverse model is the one used by
# rgbd_captures/precision_box_detector/precision_box_detector/camera_model.py
# (_distort_normalized / _undistort_normalized, fixed-point inverse); the
# optional rational terms k4..k6 (8-coefficient D) reduce to exactly that
# model when they are zero, which is the case for the Orbbec CameraInfo.

_SUPPORTED_DISTORTION_MODELS = ("plumb_bob", "rational_polynomial")


def _camera_distortion(camera_info):
    """Return (model, d) where d is a tuple of 8 floats (k1,k2,p1,p2,k3,k4,k5,k6),
    or (model, None) when the CameraInfo carries no (non-zero) distortion."""
    model = camera_info.get("distortion_model") or "plumb_bob"
    raw = camera_info.get("d")
    if raw is None:
        raw = ()
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.any(values != 0.0):
        return str(model), None
    if model not in _SUPPORTED_DISTORTION_MODELS:
        raise ValueError(f"unsupported CameraInfo distortion_model {model!r} with non-zero d")
    if values.size not in (4, 5, 8) or not np.all(np.isfinite(values)):
        raise ValueError(f"CameraInfo distortion d has unsupported length {values.size} or non-finite values")
    padded = np.zeros(8, dtype=np.float64)
    padded[:values.size] = values
    return str(model), tuple(float(value) for value in padded)


def _distort_normalized(xn, yn, d):
    """Forward plumb_bob (+ optional rational denominator) on normalized coords."""
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    r2 = xn * xn + yn * yn
    radial = (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) / \
        (1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2)
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def _undistort_normalized(xd, yd, d, iterations=100, tol=1e-12):
    """Fixed-point inverse of _distort_normalized (same iteration as
    camera_model._undistort_normalized). Fails closed if it does not converge."""
    k1, k2, p1, p2, k3, k4, k5, k6 = d
    xd = np.asarray(xd, dtype=np.float64)
    yd = np.asarray(yd, dtype=np.float64)
    xn = np.array(xd, copy=True)
    yn = np.array(yd, copy=True)
    for _ in range(iterations):
        r2 = xn * xn + yn * yn
        radial = (1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) / \
            (1.0 + k4 * r2 + k5 * r2 * r2 + k6 * r2 * r2 * r2)
        xn_new = (xd - 2.0 * p1 * xn * yn - p2 * (r2 + 2.0 * xn * xn)) / radial
        yn_new = (yd - p1 * (r2 + 2.0 * yn * yn) - 2.0 * p2 * xn * yn) / radial
        step = max(float(np.max(np.abs(xn_new - xn), initial=0.0)),
                   float(np.max(np.abs(yn_new - yn), initial=0.0)))
        xn, yn = xn_new, yn_new
        if step < tol:
            break
    check_x, check_y = _distort_normalized(xn, yn, d)
    residual = max(float(np.max(np.abs(check_x - xd), initial=0.0)),
                   float(np.max(np.abs(check_y - yd), initial=0.0)))
    if not np.isfinite(residual) or residual > 1e-9:
        raise ValueError(f"lens undistortion did not converge (normalized residual {residual:.3g})")
    return xn, yn


def _max_undistorted_radius2(fx, fy, cx, cy, image_width, image_height, d):
    """Largest undistorted r^2 over the image border; used to reject cloud points
    far outside the field of view where the distortion polynomial may fold."""
    us = np.array([0.0, image_width, 0.0, image_width, cx, cx, 0.0, image_width])
    vs = np.array([0.0, 0.0, image_height, image_height, 0.0, image_height, cy, cy])
    xn, yn = _undistort_normalized((us - cx) / fx, (vs - cy) / fy, d)
    return float(np.max(xn * xn + yn * yn))


# Points are projected in fixed-size chunks: every quantity below is a pure
# per-point expression, so chunking changes no value or order (outputs are
# bit-identical to one full-array pass) while keeping the ~20 temporaries of
# the distortion model cache-resident instead of streaming 1.7 M-row arrays.
_PROJECT_CHUNK = 1 << 16


def _project_chunk(x, y, z, fx, fy, cx, cy, image_width, image_height,
                   distortion, r2_limit_fn):
    """Project one chunk; returns (xyz_valid_in_image (M,3), u (M,), v (M,),
    n_valid). Same arithmetic as the historical full-array implementation.
    ``r2_limit_fn()`` is only evaluated when the chunk has valid points (the
    full-array code raised "no valid points" before computing it)."""
    xyz = np.column_stack((
        np.asarray(x, dtype=np.float64),
        np.asarray(y, dtype=np.float64),
        np.asarray(z, dtype=np.float64),
    ))
    valid = np.isfinite(xyz).all(axis=1) & (xyz[:, 2] > 0.0)
    xyz = xyz[valid]
    if distortion is None:
        u = fx * xyz[:, 0] / xyz[:, 2] + cx
        v = fy * xyz[:, 1] / xyz[:, 2] + cy
        in_image = (u >= 0.0) & (u < image_width) & (v >= 0.0) & (v < image_height)
    else:
        if xyz.shape[0] == 0:
            return xyz, np.empty(0), np.empty(0), 0
        r2_limit = r2_limit_fn()
        xn = xyz[:, 0] / xyz[:, 2]
        yn = xyz[:, 1] / xyz[:, 2]
        xd, yd = _distort_normalized(xn, yn, distortion)
        u = fx * xd + cx
        v = fy * yd + cy
        in_image = (xn * xn + yn * yn <= r2_limit) & \
            (u >= 0.0) & (u < image_width) & (v >= 0.0) & (v < image_height)
    return xyz[in_image], u[in_image], v[in_image], int(xyz.shape[0])


def _iter_projected_chunks(points_flat, fx, fy, cx, cy, image_width, image_height,
                           distortion=None):
    """Yield (xyz, u, v, n_valid) per chunk of ``points_flat`` in point order."""
    cache = []

    def r2_limit_fn():
        if not cache:
            cache.append(1.5 * _max_undistorted_radius2(fx, fy, cx, cy, image_width,
                                                        image_height, distortion))
        return cache[0]

    fx_, fy_, fz_ = points_flat["x"], points_flat["y"], points_flat["z"]
    n = int(np.shape(fx_)[0]) if np.ndim(fx_) else 1
    for start in range(0, max(n, 1), _PROJECT_CHUNK):
        stop = start + _PROJECT_CHUNK
        yield _project_chunk(fx_[start:stop], fy_[start:stop], fz_[start:stop],
                             fx, fy, cx, cy, image_width, image_height, distortion,
                             r2_limit_fn)


def _project_valid_cloud(points_flat, fx, fy, cx, cy, image_width, image_height, distortion=None):
    parts = list(_iter_projected_chunks(points_flat, fx, fy, cx, cy, image_width,
                                        image_height, distortion=distortion))
    if sum(part[3] for part in parts) == 0:
        raise ValueError("point cloud contains no finite positive-depth points")
    return (np.concatenate([part[0] for part in parts], axis=0),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]))


def _polygon_orientation(polygon):
    area = 0.5 * np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1) -
                        np.roll(polygon[:, 0], -1) * polygon[:, 1])
    if abs(area) < 1e-6:
        raise ValueError("detected carton polygon is degenerate")
    orientation = 1.0 if area > 0.0 else -1.0
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = end - start
        if math.hypot(dx, dy) < 1e-6:
            raise ValueError("detected carton polygon has duplicate corners")
    return orientation


def _inside_eroded_convex_polygon_oriented(u, v, polygon, inset_px, orientation):
    inside = np.ones(u.shape, dtype=bool)
    for start, end in zip(polygon, np.roll(polygon, -1, axis=0)):
        dx, dy = end - start
        length = math.hypot(dx, dy)
        signed_distance = orientation * (dx * (v - start[1]) - dy * (u - start[0])) / length
        inside &= signed_distance >= inset_px
    return inside


def _inside_eroded_convex_polygon(u, v, polygon, inset_px):
    orientation = _polygon_orientation(polygon)
    return _inside_eroded_convex_polygon_oriented(u, v, polygon, inset_px, orientation)


def _surface_candidates(points_flat, polygon, inset_px, fx, fy, cx, cy, image_width,
                        image_height, distortion=None):
    """Fused, chunked ``_project_valid_cloud`` + ``_inside_eroded_convex_polygon``:
    returns (surface_xyz, surface_u, surface_v), identical to
    ``xyz[mask], u[mask], v[mask]`` of the two-step form (same errors, same
    order: no-valid-points first, then polygon validity)."""
    try:
        orientation = _polygon_orientation(polygon)
        polygon_error = None
    except ValueError as error:
        orientation, polygon_error = 1.0, error
    xyz_parts, u_parts, v_parts = [], [], []
    n_valid = 0
    for xyz, u, v, n in _iter_projected_chunks(points_flat, fx, fy, cx, cy, image_width,
                                               image_height, distortion=distortion):
        n_valid += n
        if polygon_error is not None:
            continue
        mask = _inside_eroded_convex_polygon_oriented(u, v, polygon, inset_px, orientation)
        xyz_parts.append(xyz[mask])
        u_parts.append(u[mask])
        v_parts.append(v[mask])
    if n_valid == 0:
        raise ValueError("point cloud contains no finite positive-depth points")
    if polygon_error is not None:
        raise polygon_error
    return (np.concatenate(xyz_parts, axis=0), np.concatenate(u_parts),
            np.concatenate(v_parts))


def _fit_plane_ransac(points, *, threshold_m, iterations, min_inliers, min_ratio):
    if points.shape[0] < min_inliers:
        raise ValueError(f"only {points.shape[0]} carton-surface candidates; need {min_inliers}")
    generator = np.random.default_rng(0)
    best_mask = None
    best_count = 0
    best_error = math.inf
    for _ in range(iterations):
        first, second, third = points[generator.choice(points.shape[0], 3, replace=False)]
        normal = np.cross(second - first, third - first)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        offset = -float(np.dot(normal, first))
        residuals = np.abs(points @ normal + offset)
        inliers = residuals <= threshold_m
        count = int(np.count_nonzero(inliers))
        error = float(np.mean(residuals[inliers])) if count else math.inf
        if count > best_count or (count == best_count and error < best_error):
            best_mask, best_count, best_error = inliers, count, error
    if best_mask is None or best_count < min_inliers:
        raise ValueError(f"plane RANSAC found only {best_count} inliers")
    if best_count / points.shape[0] < min_ratio:
        raise ValueError(f"plane inlier ratio {best_count / points.shape[0]:.3f} is below {min_ratio:.3f}")
    inlier_points = points[best_mask]
    centroid = np.mean(inlier_points, axis=0)
    _, _, axes = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = axes[-1]
    normal /= np.linalg.norm(normal)
    # Point toward the camera; the eye-on-base TF maps this to the carton outward normal.
    if np.dot(normal, centroid) > 0.0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(inlier_points @ normal + offset)
    return normal, offset, float(np.sqrt(np.mean(residuals ** 2))), best_mask


def recover_surface_points(points_flat, seam_points, camera_info, *, surface_inset_px=16.0,
                           support_radius_px=64.0, min_local_support=6,
                           plane_threshold_m=0.005, plane_iterations=100,
                           min_plane_inliers=80, min_plane_ratio=0.5):
    required = ("back_left", "back_right", "front_left", "front_right", "left_mid", "right_mid")
    missing = [name for name in required if name not in seam_points]
    if missing:
        raise ValueError(f"detector output is incomplete; missing {missing}")
    image_width = int(camera_info["width"])
    image_height = int(camera_info["height"])
    distortion_model, distortion = _camera_distortion(camera_info)
    if distortion is None:
        fx, fy, cx, cy = _camera_intrinsics(camera_info)
        intrinsics_source = "p" if np.asarray(camera_info.get("p", ()), dtype=np.float64).size == 12 else "k"
    else:
        # Distorted (raw) image: K is the camera matrix that pairs with D.
        matrix = np.asarray(camera_info.get("k", ()), dtype=np.float64)
        if matrix.size != 9 or not np.all(np.isfinite(matrix)) or matrix[0] <= 0.0 or matrix[4] <= 0.0:
            raise ValueError("capture manifest CameraInfo has distortion but no valid K matrix")
        fx, fy, cx, cy = float(matrix[0]), float(matrix[4]), float(matrix[2]), float(matrix[5])
        intrinsics_source = "k"
    polygon = np.asarray([
        seam_points["back_left"], seam_points["back_right"],
        seam_points["front_right"], seam_points["front_left"],
    ], dtype=np.float64)
    if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
        raise ValueError("detected carton corners are invalid")
    if np.any(polygon[:, 0] < 0.0) or np.any(polygon[:, 0] >= image_width) or \
            np.any(polygon[:, 1] < 0.0) or np.any(polygon[:, 1] >= image_height):
        raise ValueError("detected carton corners lie outside the captured RGB image")
    surface_xyz, surface_u, surface_v = _surface_candidates(
        points_flat, polygon, surface_inset_px, fx, fy, cx, cy, image_width, image_height,
        distortion=distortion)
    normal, offset, rms, ransac_mask = _fit_plane_ransac(
        surface_xyz, threshold_m=plane_threshold_m, iterations=plane_iterations,
        min_inliers=min_plane_inliers, min_ratio=min_plane_ratio)
    inlier_xyz = surface_xyz[ransac_mask]
    inlier_u = surface_u[ransac_mask]
    inlier_v = surface_v[ransac_mask]
    results = {}
    for name in required:
        u, v = (float(value) for value in seam_points[name])
        if not (0.0 <= u < image_width and 0.0 <= v < image_height):
            raise ValueError(f"{name} lies outside captured RGB image")
        local = (inlier_u - u) ** 2 + (inlier_v - v) ** 2 <= support_radius_px ** 2
        support_count = int(np.count_nonzero(local))
        if support_count < min_local_support:
            raise ValueError(
                f"{name} has only {support_count} measured carton-surface support points within "
                f"{support_radius_px:.1f} pixels")
        xn, yn = (u - cx) / fx, (v - cy) / fy
        if distortion is not None:
            xn, yn = (float(value) for value in _undistort_normalized(xn, yn, distortion))
        ray = np.array([xn, yn, 1.0], dtype=np.float64)
        denominator = float(np.dot(normal, ray))
        if abs(denominator) < 1e-9:
            raise ValueError(f"{name} pixel ray is parallel to the measured carton surface")
        depth = -offset / denominator
        if not np.isfinite(depth) or depth <= 0.0:
            raise ValueError(f"{name} ray-plane intersection is behind the camera")
        point = ray * depth
        results[name] = {
            "u": u, "v": v, "x": float(point[0]), "y": float(point[1]), "z": float(point[2]),
            "valid": True, "depth_source": "measured_carton_plane_ray_intersection",
            "local_surface_support": support_count,
        }
    metadata = {
        "source_frame": camera_info["header"]["frame_id"],
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "source": intrinsics_source},
        "distortion": {
            "applied": distortion is not None,
            "model": distortion_model,
            "d": [float(value) for value in (camera_info.get("d") or ())],
            "method": ("undistort named pixels (fixed-point inverse, camera_model.pixel_to_ray "
                       "compatible); project cloud with forward distortion")
            if distortion is not None else "pinhole (no non-zero distortion coefficients)",
        },
        "surface_inset_px": surface_inset_px, "plane_threshold_m": plane_threshold_m,
        "plane_candidate_count": int(surface_xyz.shape[0]), "plane_inlier_count": int(inlier_xyz.shape[0]),
        "plane_inlier_ratio": float(inlier_xyz.shape[0] / surface_xyz.shape[0]),
        "plane_normal": [float(value) for value in normal], "plane_offset_m": float(offset),
        "plane_rms_m": rms,
    }
    return results, metadata


def _print_result(name, result):
    if "reprojected_u" in result:
        print(f"[{name}] 目標像素 ({result.get('u')}, {result.get('v')})"
              f"　反投影最近點於 ({result['reprojected_u']:.1f}, {result['reprojected_v']:.1f})")
    else:
        print(f"[{name}] 像素 ({result.get('u')}, {result.get('v')})"
              + (f"　←　原始查詢像素 {tuple(result['requested_pixel'].values())}"
                 if "requested_pixel" in result else ""))
    if result["valid"]:
        print(f"    X = {result['x']:.4f} m, Y = {result['y']:.4f} m, Z = {result['z']:.4f} m")
    else:
        print("    深度無效，找不到可用的替代點")
    if "rgb" in result:
        print(f"    RGB = {result['rgb']}")
    if "note" in result:
        print(f"    備註: {result['note']}")
    print()


_CORNER_KEYS = ("back_left", "back_right", "front_left", "front_right")


def _load_seam_points_from_json(seam_path):
    """讀 detect_tape_seam.py 輸出的 JSON，回傳 {"left_mid": [x,y], "right_mid": [x,y],
    "back_left": [x,y], "back_right": [x,y], "front_left": [x,y], "front_right": [x,y]}。

    detect_tape_seam.py 實際寫出的欄位是 seam_line: [[lx,ly],[rx,ry]]（沒有
    left_mid/right_mid 這兩個獨立欄位），這裡統一轉換成 left_mid/right_mid
    兩個鍵，方便跟 --image 偵測模式共用同一套查詢流程。back_left/back_right/
    front_left/front_right 是畫面上兩條紅線（back_edge、front_edge）的端點，
    detect_tape_seam.py 較新版本才會輸出，舊的 JSON 檔沒有這幾個欄位的話就
    跳過，不影響 left_mid/right_mid 照常查詢。
    """
    with open(seam_path, "r", encoding="utf-8") as f:
        seam_data = json.load(f)

    points = {}
    if "seam_line" in seam_data and len(seam_data["seam_line"]) == 2:
        left, right = seam_data["seam_line"]
        points["left_mid"] = left
        points["right_mid"] = right
    elif "left_mid" in seam_data and "right_mid" in seam_data:
        points["left_mid"] = seam_data["left_mid"]
        points["right_mid"] = seam_data["right_mid"]
    else:
        print(
            f"警告: {seam_path} 裡找不到 'seam_line'（或 'left_mid'/'right_mid'），"
            "無法取得查詢座標",
            file=sys.stderr,
        )

    missing_corners = [key for key in _CORNER_KEYS if key not in seam_data]
    if missing_corners and points:
        print(
            f"提醒: {seam_path} 裡沒有 {missing_corners}（可能是舊版 detect_tape_seam.py "
            "存的 JSON），只會查 left_mid/right_mid，紅線端點的部分請重新跑一次偵測",
            file=sys.stderr,
        )
    for key in _CORNER_KEYS:
        if key in seam_data:
            points[key] = seam_data[key]

    return points


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="query_depth_at_pixel.py",
        description="從 PCD 點雲中，依像素座標查詢對應的 3D 座標（相機座標系）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--pcd", required=True, help="PCD 檔案路徑")
    parser.add_argument(
        "--capture-manifest", default=None, metavar="MANIFEST_PATH",
        help="rgbd_capture/v1 manifest；與 learned seam JSON 一起使用時啟用量測紙箱頂面平面 recovery",
    )
    parser.add_argument(
        "--pixel", nargs=2, type=int, metavar=("U", "V"),
        help="要查詢的單一像素座標，例如 --pixel 649 159",
    )
    parser.add_argument(
        "--seam-json",
        help="seam JSON saved beforehand (left_mid/right_mid + corners); its points are queried",
    )
    parser.add_argument(
        "--max-search-radius", type=int, default=15,
        help="[organized 模式] 當目標像素深度無效時，往外搜尋最近有效深度點的最大半徑（像素），預設 15",
    )
    parser.add_argument(
        "--fx", type=float, default=None, help="[unorganized 模式必填] 彩色相機焦距 fx（像素）"
    )
    parser.add_argument(
        "--fy", type=float, default=None, help="[unorganized 模式必填] 彩色相機焦距 fy（像素）"
    )
    parser.add_argument(
        "--cx", type=float, default=None, help="[unorganized 模式必填] 彩色相機主點 cx（像素）"
    )
    parser.add_argument(
        "--cy", type=float, default=None, help="[unorganized 模式必填] 彩色相機主點 cy（像素）"
    )
    parser.add_argument(
        "--max-pixel-distance", type=float, default=3.0,
        help="[unorganized 模式] 反投影後容許的最大像素誤差，預設 3.0",
    )
    parser.add_argument("--output-json", help="將查詢結果另存成 JSON 檔（可選）")
    args = parser.parse_args(argv)

    if not args.pixel and not args.seam_json:
        parser.error("請至少指定 --pixel 或 --seam-json 其中一種查詢方式")

    try:
        header, points_flat, width, height = load_pcd(args.pcd)
    except PCDReadError as exc:
        print(f"讀取 PCD 失敗: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"已載入點雲: {args.pcd}（WIDTH={width}, HEIGHT={height}）")
    manifest = None
    if args.capture_manifest:
        try:
            manifest = load_capture_manifest(args.capture_manifest, args.pcd)
        except ValueError as error:
            print(f"capture manifest 驗證失敗: {error}", file=sys.stderr)
            sys.exit(1)
        if width != manifest["pointcloud"]["width"] or height != manifest["pointcloud"]["height"]:
            print("PCD layout 不符合 capture manifest", file=sys.stderr)
            sys.exit(1)

    organized = height > 1
    points_2d = points_flat.reshape(height, width) if organized else None
    resolved_intrinsics = None
    if organized:
        print("→ Organized 點雲，使用像素索引查詢")
    else:
        try:
            resolved_intrinsics = resolve_intrinsics(manifest, args)
        except ValueError as error:
            print(f"Unorganized 點雲內參驗證失敗: {error}", file=sys.stderr)
            sys.exit(1)
        if manifest:
            print("→ Unorganized registered cloud，使用 capture manifest 的已驗證投影與量測頂面平面")
        else:
            print(f"→ Unorganized 點雲，使用 legacy 反投影最近點搜尋（容許誤差 {args.max_pixel_distance} 像素）")
    print()

    results = {}
    seam_points = {}
    surface_metadata = None

    def do_query(u, v, key=None):
        if organized:
            edge_partner_uv = None
            if key is not None:
                partner_key = _EDGE_PARTNER.get(key)
                if partner_key and partner_key in seam_points:
                    pu, pv = seam_points[partner_key]
                    edge_partner_uv = (int(round(pu)), int(round(pv)))
            return query_pixel_with_fallback(
                points_2d, u, v, max_radius=args.max_search_radius,
                edge_partner_uv=edge_partner_uv,
            )
        return query_by_reprojection(
            points_flat, u, v, *resolved_intrinsics,
            max_pixel_distance=args.max_pixel_distance,
        )

    if args.pixel:
        u, v = args.pixel
        try:
            results["pixel"] = do_query(u, v)
        except ValueError as error:
            print(f"查詢失敗: {error}", file=sys.stderr)
            sys.exit(1)

    if args.seam_json:
        seam_path = Path(args.seam_json)
        if not seam_path.exists():
            print(f"找不到檔案: {seam_path}", file=sys.stderr)
            sys.exit(1)
        seam_points = _load_seam_points_from_json(seam_path)

    if seam_points and manifest:
        try:
            surface_results, surface_metadata = recover_surface_points(
                points_flat, seam_points, manifest["camera_info"])
        except ValueError as error:
            print(f"量測紙箱頂面 recovery 失敗: {error}", file=sys.stderr)
            sys.exit(1)
        results.update(surface_results)
    else:
        for key, (px, py) in seam_points.items():
            u = int(round(px))
            v = int(round(py))
            try:
                results[key] = do_query(u, v, key=key)
            except ValueError as error:
                print(f"查詢 {key} 失敗: {error}", file=sys.stderr)

    for name, result in results.items():
        _print_result(name, result)

    if args.output_json:
        output_path = Path(args.output_json)
        if manifest:
            output = {
                "schema": "seam_camera_geometry/v1",
                "capture": {
                    "capture_id": manifest["capture_id"],
                    "image_stamp_ns": manifest["image"]["header"]["stamp_ns"],
                    "pointcloud_stamp_ns": manifest["pointcloud"]["header"]["stamp_ns"],
                    "source_frame": manifest["pointcloud"]["header"]["frame_id"],
                    "manifest_path": str(Path(args.capture_manifest).resolve()),
                },
                "surface": surface_metadata,
                "results": results,
            }
        else:
            output = results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent,
                                         prefix=f".{output_path.name}.", suffix=".tmp", delete=False) as file:
            json.dump(output, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
            temporary_name = file.name
        try:
            os.replace(temporary_name, output_path)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise
        print(f"已將結果寫入 {output_path}")
if __name__ == "__main__":
    main()
