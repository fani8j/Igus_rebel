#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detect_tape_seam.py

延續 detect_box_edges.py 的紙箱偵測，這支程式進一步鎖定紙箱「頂面」
（top face）的四個角落，找出頂面的左邊、右邊這兩條邊，把它們的
位置相加除以 2（也就是各自邊的中點連成一條線），得出膠帶封箱縫隙
（切割軌跡）的估計位置。

流程：
  1. 用 GrabCut 在指定 ROI 內把整個紙箱從背景分割出來（跟 detect_box_edges.py
     同一套做法）。
  2. 紙箱的頂面通常比正面/側面亮（光線從上方打下來），所以在紙箱範圍內對
     灰階亮度做 Otsu 二值化，把「頂面」從「側面」分開，取面積最大的亮區塊。
  3. 對這個頂面區塊做多邊形化簡（approxPolyDP），自動搜尋合適的誤差值，
     直到化簡成 4 個角點為止（頂面在透視下應該是一個四邊形）。
  4. 用整條輪廓上落在每一邊的點分別做直線擬合，取相鄰兩邊擬合直線的交點
     取代 approxPolyDP 給的原始角點（refine_corners_by_line_fit）。這一步
     是為了修正單一角點附近如果剛好有摺痕、陰影、或分割邊界圓角化，導致
     approxPolyDP 把頂點放到偏離真正幾何角點（三個面的交會點）的位置。
  5. 直接在原圖顏色上，對四邊形每一條邊沿邊的中段（避開角點）取樣，找出
     「非頂面色 -> 頂面色」的顏色轉換點，重新擬合每條邊的直線，取相鄰兩邊
     交點再修正一次角點座標（refine_corners_by_color_edges）。這一步是為了
     處理「頂面亮度遮罩本身在某個角落附近分割得不夠乾淨」的情況——如果遮罩
     輪廓在角點附近就已經偏離真正邊界，單靠第 4 步對著這個偏離的輪廓做直線
     擬合，交點還是會偏（實測案例：亮度遮罩讓角點多算進了將近 30px 的正面
     區域，第 4 步只修正了約 20px，仍然明顯不對；改成直接找原圖顏色邊界後，
     才真正修正到位）。頂面是偏暖色（R 明顯大於 B）且夠亮，跟背景（偏中性
     色）、正面（明顯較暗）都有清楚區別，見 _is_top_face_color()。
  6. 把 4 個角點依照「畫面中位置較高（y 較小）」判定為後方邊、「位置較低」
     判定為前方邊，剩下兩條邊再依照平均 x 座標分成左邊、右邊。
  7. 分別取左邊、右邊各自的中點，兩個中點相連，就是膠帶縫隙／切割軌跡的
     估計線段。

備註：
  這個判斷「哪條邊是後方/前方，剩下哪兩條是左/右」的邏輯，是根據目前這台
  相機的拍攝角度（由上方略微傾斜往下拍，紙箱後方邊緣在畫面裡的 y 座標比
  前方邊緣小）寫的。只要相機角度、紙箱擺放方式不變，這個假設就會持續成立；
  如果之後換了完全不同的拍攝角度（例如整個倒過來拍），才需要回來調整判斷
  邏輯（可以在 --debug 模式下看每條邊被歸類的結果，確認有沒有抓對）。

  第 5 步（refine_corners_by_color_edges）如果某條邊找不到足夠的顏色轉換
  點（例如背景跟頂面顏色太接近、光線太暗），會自動保留第 4 步的角點，不會
  套用不穩定的結果，所以正常情況下不需要額外處理，但如果 --debug 觀察到
  角點看起來又跑掉了，可以先確認是不是那條邊的顏色對比不夠明顯。

  2026-09-01：auto-ROI（estimate_auto_roi）加了位置先驗，detect_seam()
  最後加了 _validate_top_face_quad_color() 做顏色合理性檢查，細節見各自
  函式的 docstring。加入原因：實測有一次箱子旁邊的橘色卷尺/紅色簽字筆/
  綠色膠帶台被橋接進候選輪廓，auto-ROI 選中的 ROI 變成 (615, 89, 901, 380)
  （寬286×高291），比對「黃金範例」frame 20260831_164853 的正確 ROI
  (665, 100, 896, 285)（寬231×高185，長寬比~1.25，緊貼箱子沒多框東西）
  明顯胖了一圈，導致 GrabCut/Otsu 把混進來的白色桌面誤判成頂面，回傳出
  完全不在箱子上的四邊形。

  2026-09-02：原本還加了 min_solidity=0.8 硬性過濾候選（理論上橋接雜物
  的候選凸包會偏大、solidity 偏低），但拿實機資料實測後發現這個門檻誤傷
  了另一組正確案例（frame_..._105447，疊放的細長箱子，正常情況 solidity
  本來就低於 0.8），所以移除了這條硬性過濾，solidity 改成只計算、只在
  --debug 圖上顯示，不拿來排除候選（見 estimate_auto_roi docstring）。
  橋接雜物這個問題目前只靠 _validate_top_face_quad_color() 把關，它的
  min_warm_ratio=0.5 也還沒拿實機資料驗證過，之後要用 --debug 印出的
  warm_ratio 數值回頭校正，尤其要確認它會不會反過來誤傷合法案例。

  2026-09-02（重要修正）：上面 09-01/09-02 那幾個修正，方向其實錯了——
  目標應該是「不管環境多亂，都要正確辨識出箱子的4個角點」，_validate_
  top_face_quad_color() 只能讓程式在抓錯的時候丟出清楚的錯誤，並不能讓
  程式在那張照片裡真的找到箱子。真正根因是箱子在 _cardboard_color_mask()
  這一步就已經跟旁邊的橘色卷尺/紅色簽字筆黏成同一個連通元件，後面不管
  怎麼篩選候選都救不回來。改用 saturation_hi 上限（見 _cardboard_
  color_mask docstring）從遮罩階段就把這些鮮豔小物排除掉，讓箱子單獨成
  一塊——這個嘗試還沒驗證過，需要拿 debug_auto_roi_mask.png 確認箱子
  本身沒被誤傷、鮮豔小物真的被排除，並確認最後能不能真的跑出箱子的
  4 個角點。

使用方式：
  python3 detect_tape_seam.py
  python3 detect_tape_seam.py --image /path/to/other_frame.png --roi 500 80 900 320
  python3 detect_tape_seam.py --debug   # 額外印出每條邊的分類過程，方便除錯

必要套件：
  pip install opencv-python-headless numpy --break-system-packages
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


DEFAULT_IMAGE = "/home/james/Igus_rebel/outputs/rgb/frame_000001_20260821_121845_735625.png"
# 針對這張照片手動框出的「中央偏上方紙箱」大略範圍 (x0, y0, x1, y1)
DEFAULT_ROI = (590, 90, 860, 300)


# ---------------------------------------------------------------------------
# 自動抓大概位置（auto ROI）：箱子移動/換角度後，不用每次手動改 --roi
# ---------------------------------------------------------------------------

def _cardboard_color_mask(image, brightness_lo=60, brightness_hi=245,
                           warmth_thresh=14, saturation_lo=20, saturation_hi=140):
    """粗略框出畫面中『偏暖色、亮度適中、有一定飽和度』的像素，用來涵蓋整個
    紙箱（不只頂面，側面較暗也要抓到），跟 _is_top_face_color() 是同一個
    邏輯精神，只是門檻放寬，因為這裡的目標是「抓大概位置」，不是精確頂面
    邊界，交給後面 GrabCut + 角點修正去做精細的部分。

    加入 HSV 飽和度下限（saturation_lo）是為了濾掉背景常見的高亮反光
    （例如白色桌面反光、金屬反光），這些點雖然也可能亮度夠、R-B 差略大於0，
    但飽和度通常偏低（趨近灰白），跟紙箱這種有明顯顏色的表面不同。

    **飽和度上限 saturation_hi（2026-09-02 加入，處理「箱子被旁邊鮮豔小物
    橋接」的根本原因）**：紙箱是偏暗黃褐色、飽和度中等的表面；橘色卷尺、
    紅色簽字筆這類鮮豔小物雖然也偏暖色、也夠亮，但飽和度明顯比紙箱高很多
    （鮮豔 vs 混濁）。之前沒有上限，導致這些鮮豔小物也被算進「箱子候選
    顏色」，跟紙箱連成同一塊輪廓，後面不管怎麼篩選候選都救不回來（一旦在
    這一步黏在一起，就沒有乾淨的「只框到箱子」的輪廓可選）。加上
    saturation_hi 從遮罩階段就把這些鮮豔物品排除掉，讓箱子在連通元件分析
    時能單獨成一塊。

    **這個門檻完全沒有拿實機資料驗證過，是純推論值**：可能誤傷紙箱本身
    （如果某個角度反光讓紙箱局部飽和度也衝到 140 以上，該部分像素會被排除，
    可能在箱子輪廓上挖出缺口甚至把箱子切成兩塊）。加了 --debug 一定要打開
    `debug_auto_roi_mask.png` 親眼確認：(1) 箱子本身有沒有被挖洞/切斷，
    (2) 橘色卷尺/紅色簽字筆有沒有真的從遮罩裡消失、跟箱子分開。這兩件事
    都要用真實圖片檢查過才能信任這個門檻，不能只看最後有沒有跑出結果就
    覺得成功。
    """
    b = image[:, :, 0].astype(np.float32)
    g = image[:, :, 1].astype(np.float32)
    r = image[:, :, 2].astype(np.float32)
    brightness = (r + g + b) / 3.0
    warmth = r - b
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)

    mask = (
        (brightness > brightness_lo) & (brightness < brightness_hi) &
        (warmth > warmth_thresh) &
        (saturation > saturation_lo) & (saturation < saturation_hi)
    ).astype(np.uint8) * 255
    return mask


def estimate_auto_roi(image, min_area_ratio=0.015, max_area_ratio=0.5,
                       min_extent=0.35, pad_ratio=0.15,
                       expected_center_ratio=(0.55, 0.24),
                       max_center_dist_ratio=0.35):
    """自動抓出畫面中紙箱大概位置的 ROI，取代手動框 --roi。

    做法：先用 _cardboard_color_mask() 抓出「偏暖色」的候選像素，做形態學
    open/close 把雜訊清掉、把箱子頂面+側面連成一塊，再找輪廓，過濾掉太小
    （雜訊）或太大（例如整張圖過曝偏暖）的候選，同時用 `extent`（輪廓面積
    / 外接矩形面積）過濾掉形狀太破碎、太細長的候選（例如人的手臂、頭髮這類
    窄長區塊，extent 通常明顯偏低）。

    **位置先驗（2026-09-01 加入）**：純靠顏色 + 面積最大來選候選，實測遇過
    誤選案例——畫面右下角椅子旁露出的木紋地板剛好也偏暖色、亮度適中、
    extent 也夠高，面積又比真正的紙箱候選大，於是被排序成第一名選走
    （auto_roi.png 上看到選中框框到椅子/地板，而不是箱子）。回頭比對三次
    正確偵測到的紙箱位置（frame_..._105447、104843、105219 三組
    tape_seam.json 裡的 roi，換算成候選中心點都落在畫面寬度中央偏左一點、
    高度上方約 1/4 處，即 expected_center_ratio=(0.55, 0.24) 附近），跟那次
    誤選的椅子候選（中心點在畫面右下方）距離明顯拉開，因此加入這個先驗：
    候選中心點跟「預期中心」的距離（除以畫面對角線長度正規化）超過
    max_center_dist_ratio 的候選會被優先排除，只在候選裡挑面積最大的那個；
    如果套用先驗後完全沒有候選留下（例如箱子這次真的擺到很偏的位置），才
    退回用全部候選挑面積最大，並印出警告提醒務必檢查 `_auto_roi.png`。

    **形狀（solidity）——只顯示不過濾（2026-09-02 修正）**：曾經試過用
    solidity（輪廓面積 / 凸包面積）當硬性過濾，理論上「紙箱被旁邊暖色小物
    橋接成一塊」會讓凸包多算進物件間的空隙、solidity 偏低，藉此擋掉這種
    候選。但拿實機資料（--debug）實測後發現 `min_solidity=0.8` 直接把一組
    真正正確的候選也判掉了（frame_..._105447 那組疊放的細長箱子，正常
    情況solidity 就低於 0.8），是會誤傷正例的硬性門檻，而且因為濾除發生在
    候選收集階段，被濾掉的候選連數值都不會印出來，沒辦法回頭校正。因此
    移除這條硬性過濾：solidity 只計算、放進 candidate 資訊、印在
    `_auto_roi.png` 上供人工參考，不拿來排除候選。真正要擋「橋接雜物→
    GrabCut/Otsu 誤判成桌面」這種情況，交給 detect_seam() 最後呼叫的
    `_validate_top_face_quad_color()`——它是直接驗證『偵測出來的頂面顏色
    像不像紙箱』，比在候選輪廓形狀上做代理猜測（solidity）更貼近問題本身，
    也不會誤傷「箱子本身形狀就不那麼規則」的合法候選。

    重要限制（務必知道）：這是顏色 + 位置為主的粗略估計，不是真正的物件
    偵測，兩層過濾可能同時失效（例如另一個物體剛好落在預期位置附近、
    面積又最大）。expected_center_ratio 是根據目前這個工作站的相機角度、
    桌面/箱子擺放習慣統計出來的經驗值，換了相機角度或箱子擺放位置差很多，
    需要跟著重新調整（可以再拿幾張新的正確偵測結果的 roi 回頭重新估算）。
    因此回傳的 ROI 只當作『大概範圍＋外擴一點邊界』，實際頂面邊界仍然是
    交給既有的 GrabCut/Otsu/角點修正流程去精算；同時強烈建議搭配 --debug
    或檢視程式存出的 `_auto_roi.png` 檢查候選框是否真的框到箱子，不要盲目
    信任自動結果，更不要只看 ROI 候選框對不對就以為整條 pipeline 沒問題
    ——一定要連 `_tape_seam.png` 的四個角點畫在哪裡都一起檢查。

    回傳 (roi, debug_info)：
      roi: (x0, y0, x1, y1)，已用 pad_ratio 外擴並裁切到畫面範圍內。
      debug_info: dict，包含所有候選框（供 --debug 畫出來檢查）、選中的
        那一個、預期中心點座標、以及是否有套用先驗過濾，方便事後確認自動
        抓的位置合不合理。
    """
    h, w = image.shape[:2]
    image_area = float(h * w)
    diag = float(np.hypot(w, h))
    expected_center = (expected_center_ratio[0] * w, expected_center_ratio[1] * h)

    mask = _cardboard_color_mask(image)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area_ratio * image_area or area > max_area_ratio * image_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        extent = area / float(bw * bh) if bw * bh > 0 else 0.0
        if extent < min_extent:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        # solidity 只計算、供 --debug 顯示參考，不拿來排除候選
        # （2026-09-02：min_solidity=0.8 硬性過濾實測誤傷了正確候選，見上方 docstring）
        solidity = area / hull_area if hull_area > 0 else 0.0
        center = (x + bw / 2.0, y + bh / 2.0)
        center_dist_ratio = float(np.hypot(center[0] - expected_center[0],
                                            center[1] - expected_center[1]) / diag)
        candidates.append({
            "bbox": (x, y, bw, bh),
            "area": area,
            "extent": extent,
            "solidity": solidity,
            "center": center,
            "center_dist_ratio": center_dist_ratio,
        })

    if not candidates:
        raise RuntimeError(
            "自動抓不到夠大、夠『實心』的暖色區塊（可能光線太暗、箱子被遮擋，"
            "或畫面裡找不到符合紙箱顏色特徵的物體），請改用 --roi 手動指定，"
            "或用 --debug 檢查 auto_roi_mask.png 確認顏色遮罩有沒有抓到箱子。"
        )

    # 位置先驗：優先只在「離預期中心夠近」的候選裡挑面積最大的（見上方
    # docstring）。全部都被先驗排除時才退回用全部候選，並提醒使用者檢查。
    within_prior = [c for c in candidates if c["center_dist_ratio"] <= max_center_dist_ratio]
    used_fallback = not within_prior
    if used_fallback:
        print(
            "警告: 沒有候選落在位置先驗範圍內（max_center_dist_ratio="
            f"{max_center_dist_ratio}），退回用全部候選挑面積最大，"
            "請務必檢查存出的 *_auto_roi.png 確認有沒有抓對。",
            file=sys.stderr,
        )
    pool = candidates if used_fallback else within_prior

    # 同一個先驗範圍內，面積最大的候選當作箱子（見上方 docstring 的驗證說明與限制）
    pool.sort(key=lambda d: d["area"], reverse=True)
    chosen = pool[0]
    x, y, bw, bh = chosen["bbox"]

    pad_x = int(bw * pad_ratio)
    pad_y = int(bh * pad_ratio)
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(w, x + bw + pad_x)
    y1 = min(h, y + bh + pad_y)

    debug_info = {
        "mask": mask_clean,
        "candidates": candidates,
        "chosen": chosen,
        "roi": (x0, y0, x1, y1),
        "expected_center": expected_center,
        "max_center_dist_ratio": max_center_dist_ratio,
        "used_fallback": used_fallback,
    }
    return (x0, y0, x1, y1), debug_info


def segment_box_mask(roi_img, inset_ratio=0.08, grabcut_iters=5):
    """用 GrabCut 分割出紙箱前景遮罩（回傳 0/255 的 mask，跟 roi_img 同大小）。"""
    h, w = roi_img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    margin_x = max(1, int(w * inset_ratio))
    margin_y = max(1, int(h * inset_ratio))
    init_rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    cv2.grabCut(roi_img, mask, init_rect, bgd_model, fgd_model,
                grabcut_iters, cv2.GC_INIT_WITH_RECT)

    box_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_OPEN, kernel)
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, kernel)
    return box_mask


def segment_top_face_mask(roi_img, box_mask):
    """在紙箱遮罩範圍內，用亮度 Otsu 門檻分出較亮的頂面。"""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    box_pixels = gray[box_mask > 0]
    if box_pixels.size == 0:
        raise RuntimeError("紙箱遮罩是空的，無法判斷頂面亮度門檻")

    otsu_thresh, _ = cv2.threshold(
        box_pixels.reshape(-1, 1).astype(np.uint8), 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )

    top_mask = np.where((gray >= otsu_thresh) & (box_mask > 0), 255, 0).astype(np.uint8)
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return top_mask


def find_quad_corners(mask, min_area_ratio=0.05):
    """在遮罩裡找最大輪廓，搜尋合適的 approxPolyDP 誤差值，化簡成 4 個角點。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("找不到頂面的輪廓，請確認 ROI 有把紙箱頂面框進去")

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    mask_area = (mask > 0).sum()
    if mask_area == 0 or area < min_area_ratio * mask_area:
        raise RuntimeError(
            f"頂面區塊太小（面積比例 {area / max(mask_area, 1):.1%}），"
            "可能亮度門檻沒有正確分出頂面，請調整 ROI 或改用 --debug 檢查中間結果。"
        )

    peri = cv2.arcLength(contour, True)
    for eps_ratio in np.arange(0.01, 0.08, 0.005):
        approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
        if len(approx) == 4:
            return approx.reshape(-1, 2).astype(float), contour

    raise RuntimeError(
        "頂面輪廓化簡不出乾淨的四邊形（可能被雜物遮擋或頂面亮度分割不乾淨），"
        "請用 --debug 檢查 top_face_mask，或調整 ROI。"
    )


def refine_corners_by_line_fit(contour, corners, max_correction_ratio=0.5):
    """用整條輪廓上落在每一邊的點做直線擬合，取相鄰兩邊擬合直線的交點，取代
    approxPolyDP 給的原始頂點座標，讓角點定位更準。

    背景：如果某個角點附近的輪廓因為分割不夠乾淨（例如摺痕、陰影、GrabCut
    邊界圓角化）而稍微鼓起或凹陷，approxPolyDP 化簡出的頂點就會落在那個
    鼓起處，而不是紙箱真正的幾何角點（三個面的交會點）。這裡改用「每一邊
    的整條輪廓線去擬合直線，再取相鄰兩邊擬合直線的交點」，因為單一角點附近
    的雜訊只佔整條邊上一小段點，直線擬合會被其餘大部分正確的點主導，交點
    自然就會落回真正的角點，而不是被局部雜訊拉走。

    corners: find_quad_corners() 回傳、依輪廓走向排列的 4 個角點（座標須為
        contour 上實際存在的點，這是 approxPolyDP 的性質，一定成立）。
    max_correction_ratio: 修正量相對於輪廓包圍盒對角線長度的上限比例，
        超過就視為擬合失敗（例如剛好遇到兩邊近乎平行的退化情況），改回
        使用原始角點，避免修正結果暴走到畫面外或明顯不合理的位置。
    """
    contour_pts = contour.reshape(-1, 2).astype(np.float64)
    diag = np.linalg.norm(contour_pts.max(axis=0) - contour_pts.min(axis=0))
    max_correction = max(5.0, diag * max_correction_ratio)

    # 找出每個角點在輪廓陣列裡的索引（approxPolyDP 的輸出點必定是原輪廓上的點）
    indices = []
    for corner in corners:
        dists = np.sum((contour_pts - corner) ** 2, axis=1)
        indices.append(int(np.argmin(dists)))

    lines = []
    for i in range(4):
        start_idx, end_idx = indices[i], indices[(i + 1) % 4]
        if start_idx <= end_idx:
            segment = contour_pts[start_idx:end_idx + 1]
        else:
            segment = np.vstack([contour_pts[start_idx:], contour_pts[:end_idx + 1]])
        if len(segment) < 2:
            segment = np.array([contour_pts[start_idx], contour_pts[end_idx]])
        vx, vy, x0, y0 = cv2.fitLine(
            segment.reshape(-1, 1, 2).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        lines.append((float(vx), float(vy), float(x0), float(y0)))

    def intersect(line_a, line_b):
        vx1, vy1, x1, y1 = line_a
        vx2, vy2, x2, y2 = line_b
        mat = np.array([[vx1, -vx2], [vy1, -vy2]])
        rhs = np.array([x2 - x1, y2 - y1])
        det = np.linalg.det(mat)
        if abs(det) < 1e-8:
            return None  # 相鄰兩邊幾乎平行，交點不穩定
        t, _ = np.linalg.solve(mat, rhs)
        return np.array([x1 + t * vx1, y1 + t * vy1])

    refined = corners.copy().astype(np.float64)
    for i in range(4):
        prev_line = lines[(i - 1) % 4]
        curr_line = lines[i]
        pt = intersect(prev_line, curr_line)
        if pt is not None and np.linalg.norm(pt - corners[i]) <= max_correction:
            refined[i] = pt
    return refined


def _is_top_face_color(pixel_bgr, brightness_thresh=130.0, warmth_thresh=20.0):
    """判斷一個 BGR 像素是否符合「頂面紙箱色」的特徵：偏暖色（R 明顯大於 B）
    且夠亮。用來直接在原圖上找頂面邊界，而不是依賴 GrabCut/Otsu 遮罩。

    背景（灰色系的桌面、家電）通常偏中性色（R、B 差不多），正面則因為
    光線角度而明顯偏暗，兩者都不會同時滿足「暖色 + 夠亮」，可以用這組
    門檻把頂面跟背景、正面分開。
    """
    b, g, r = float(pixel_bgr[0]), float(pixel_bgr[1]), float(pixel_bgr[2])
    brightness = (r + g + b) / 3.0
    warmth = r - b
    return brightness > brightness_thresh and warmth > warmth_thresh


def _find_edge_transition_points(roi_img, p1, p2, outward_normal, margin_ratio=0.18,
                                  num_samples=24, search_range=22, sustain=3):
    """沿著一條邊 (p1 -> p2) 的方向取多個取樣點，每個取樣點沿著邊的法線方向，
    由外（outward_normal 方向）往內掃描，找出顏色從「非頂面」變成「頂面」
    且持續 sustain 個像素的轉換點座標。

    margin_ratio 讓取樣點避開邊的兩端（角點附近），因為角點附近的顏色轉換
    本來就比較不乾淨（三個面交會、陰影、高光都在那附近），邊的中段取樣
    才能準確反映這條邊真正的方向。
    """
    h, w = roi_img.shape[:2]
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    edge_vec = p2 - p1
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-6:
        return []
    normal = np.asarray(outward_normal, dtype=np.float64)
    normal = normal / (np.linalg.norm(normal) + 1e-9)

    points = []
    for i in range(num_samples):
        t = margin_ratio + (1 - 2 * margin_ratio) * i / max(1, num_samples - 1)
        base = p1 + edge_vec * t
        transition = None
        run = 0
        run_start_d = None
        for d in range(search_range, -search_range - 1, -1):  # 由外往內掃描
            sample = base + normal * d
            x, y = int(round(sample[0])), int(round(sample[1]))
            if x < 0 or y < 0 or x >= w or y >= h:
                run = 0
                run_start_d = None
                continue
            if _is_top_face_color(roi_img[y, x]):
                if run == 0:
                    run_start_d = d
                run += 1
                if run >= sustain:
                    transition = base + normal * run_start_d
                    break
            else:
                run = 0
                run_start_d = None
        if transition is not None:
            points.append(transition)
    return points


def _fit_line_robust(points, max_iters=5, outlier_thresh=4.0):
    """對一組 2D 點做穩健直線擬合（迭代剔除離群點），回傳 cv2.fitLine 格式的
    (vx, vy, x0, y0)：單位方向向量 + 直線上一點。"""
    pts = np.array(points, dtype=np.float64)
    if len(pts) < 2:
        return None
    vx = vy = x0 = y0 = None
    for _ in range(max_iters):
        vx, vy, x0, y0 = cv2.fitLine(
            pts.reshape(-1, 1, 2).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()
        normal = np.array([-vy, vx])
        base = np.array([x0, y0])
        resid = (pts - base) @ normal
        keep = np.abs(resid) < outlier_thresh
        if keep.sum() == len(pts) or keep.sum() < 2:
            break
        pts = pts[keep]
    return (float(vx), float(vy), float(x0), float(y0))


def refine_corners_by_color_edges(roi_img, corners, max_correction_ratio=0.6, min_points=4):
    """直接用原圖顏色（而非遮罩輪廓）重新定位頂面四個角點，取代/補強
    refine_corners_by_line_fit()。

    背景：refine_corners_by_line_fit() 是對「遮罩輪廓」做直線擬合，如果頂面
    亮度遮罩（segment_top_face_mask，Otsu 門檻）在某個角點附近分割得不夠
    乾淨——例如那個角落頂面和側面的亮度太接近，或者剛好有陰影、反光——
    輪廓本身在那附近就已經偏離真正的邊界，用這種已經偏離的輪廓去擬合直線，
    交點自然還是偏的（實測案例：一個角點被輪廓擬合修正了約 20px，但實際
    偏差超過 30px，肉眼看仍然明顯不對）。

    改進做法：不管遮罩／輪廓，直接在原圖上找顏色邊界。紙箱頂面是偏暖色
    （R 明顯大於 B）且夠亮，跟背景（偏中性色）、正面（明顯較暗）都有清楚
    區別（見 _is_top_face_color）。對四邊形每一條邊，沿邊的中段（避開角點）
    取多個取樣點，每個取樣點往邊的外側到內側方向掃描，找出「非頂面色 ->
    頂面色」的轉換點，這些轉換點做穩健直線擬合，就是這條邊真正的位置。
    相鄰兩條邊的擬合直線交點，就是修正後的角點。

    如果某條邊找到的轉換點太少（< min_points，通常代表這條邊本身對比不夠
    明顯，或緊鄰另一個複雜背景），或者修正量超過 max_correction_ratio，就
    保留原始角點，不套用不穩定的結果。
    """
    n = len(corners)
    centroid = corners.mean(axis=0)
    diag = np.linalg.norm(corners.max(axis=0) - corners.min(axis=0))
    max_correction = max(6.0, diag * max_correction_ratio)

    lines = []
    for i in range(n):
        p1, p2 = corners[i], corners[(i + 1) % n]
        edge_vec = p2 - p1
        raw_normal = np.array([-edge_vec[1], edge_vec[0]])
        mid = (p1 + p2) / 2
        if np.dot(raw_normal, mid - centroid) < 0:
            raw_normal = -raw_normal  # 確保法線方向朝外（遠離四邊形中心）
        pts = _find_edge_transition_points(roi_img, p1, p2, raw_normal)
        lines.append(_fit_line_robust(pts) if len(pts) >= min_points else None)

    def intersect(line_a, line_b):
        if line_a is None or line_b is None:
            return None
        vx1, vy1, x1, y1 = line_a
        vx2, vy2, x2, y2 = line_b
        mat = np.array([[vx1, -vx2], [vy1, -vy2]])
        rhs = np.array([x2 - x1, y2 - y1])
        det = np.linalg.det(mat)
        if abs(det) < 1e-8:
            return None
        t, _ = np.linalg.solve(mat, rhs)
        return np.array([x1 + t * vx1, y1 + t * vy1])

    refined = corners.copy().astype(np.float64)
    for i in range(n):
        prev_line = lines[(i - 1) % n]
        curr_line = lines[i]
        pt = intersect(prev_line, curr_line)
        if pt is not None and np.linalg.norm(pt - corners[i]) <= max_correction:
            refined[i] = pt
    return refined


def order_quad_and_find_seam(corners):
    """將 4 個角點排序，判斷後/前/左/右邊，回傳左右邊中點連成的縫隙線段，
    以及後/前邊（畫面上呈紅色的那兩條邊，見 draw_visualization：整個四邊形
    先畫成紅色，left_edge/right_edge 再疊畫成綠色蓋掉，剩下沒被蓋掉、還是
    紅色的就是 back_edge 跟 front_edge）各自的兩個端點。

    因為四邊形每個角點都同時是「後/前邊」跟「左/右邊」其中一條的端點
    （四邊形相鄰邊共用角點的性質），back_edge 的兩個端點裡，一個一定也是
    left_edge 的端點、另一個一定也是 right_edge 的端點，front_edge 同理。
    依此把 back_edge/front_edge 的端點各自標成 back_left/back_right、
    front_left/front_right，方便後續程式（例如查這四個點的深度）直接取用，
    不用自己再去猜「紅線的兩個端點分別對應哪一側」。

    回傳 dict: corners_ordered（依序 4 點）、back_edge、front_edge、
    left_edge、right_edge、seam_line = (left_mid, right_mid)、
    back_left、back_right、front_left、front_right
    """
    centroid = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    order = np.argsort(angles)
    ordered = corners[order]

    edges_idx = [(i, (i + 1) % 4) for i in range(4)]
    edges = [(ordered[i], ordered[j]) for i, j in edges_idx]
    mean_ys = [(ordered[i][1] + ordered[j][1]) / 2 for i, j in edges_idx]
    back_idx = int(np.argmin(mean_ys))   # 畫面中位置較高（y 較小）＝紙箱後方邊
    front_idx = int(np.argmax(mean_ys))  # 畫面中位置較低（y 較大）＝紙箱前方邊（跟正面交界）
    remaining = [i for i in range(4) if i not in (back_idx, front_idx)]

    e1, e2 = edges[remaining[0]], edges[remaining[1]]
    mean_x1 = (e1[0][0] + e1[1][0]) / 2
    mean_x2 = (e2[0][0] + e2[1][0]) / 2
    if mean_x1 <= mean_x2:
        left_edge, right_edge = e1, e2
        left_edge_idx, right_edge_idx = edges_idx[remaining[0]], edges_idx[remaining[1]]
    else:
        left_edge, right_edge = e2, e1
        left_edge_idx, right_edge_idx = edges_idx[remaining[1]], edges_idx[remaining[0]]

    left_mid = (left_edge[0] + left_edge[1]) / 2
    right_mid = (right_edge[0] + right_edge[1]) / 2

    def split_by_side(edge_idx):
        """給一條邊的兩個角點索引，回傳 (跟 left_edge 共用端點的那個角點,
        跟 right_edge 共用端點的那個角點)。四邊形性質保證恰好各一個。"""
        a, b = edge_idx
        if a in left_edge_idx:
            return ordered[a], ordered[b]
        return ordered[b], ordered[a]

    back_left, back_right = split_by_side(edges_idx[back_idx])
    front_left, front_right = split_by_side(edges_idx[front_idx])

    return {
        "corners_ordered": ordered,
        "back_edge": edges[back_idx],
        "front_edge": edges[front_idx],
        "left_edge": left_edge,
        "right_edge": right_edge,
        "seam_line": (left_mid, right_mid),
        "back_left": back_left,
        "back_right": back_right,
        "front_left": front_left,
        "front_right": front_right,
    }


def _validate_top_face_quad_color(roi_img, corners, min_warm_ratio=0.5,
                                   brightness_thresh=130.0, warmth_thresh=10.0,
                                   num_samples=300):
    """在把偵測結果回傳出去之前，抽樣檢查『頂面』四邊形內部的顏色是不是真的
    像紙箱頂面（暖色、有一定亮度），避免 segment_top_face_mask() 的 Otsu
    亮度門檻在 ROI 混進其他更亮的東西時（例如白色桌面），誤把桌面/背景當
    頂面，卻完全沒有任何錯誤訊息地回傳出去。

    背景（2026-09-01 加入）：實測案例裡，auto-ROI 選中的候選因為跟旁邊
    暖色小物被橋接成一塊，外接矩形往下多框了一段白色桌面；GrabCut/Otsu
    在這種混雜輸入下，切出來「最亮的一塊」變成桌面而不是紙箱頂面（桌面比
    紙箱頂面亮很多），回傳的四個角點因此完全落在桌面上，但程式本身沒有
    任何機制發現這件事，只是默默印出一組看起來正常、實際上錯誤的座標。

    做法：在四邊形內部均勻抽樣一批像素，用跟 _is_top_face_color() 相同的
    「暖色 + 夠亮」判斷邏輯統計符合比例；比例低於 min_warm_ratio 就視為
    疑似抓到背景/桌面，丟出 RuntimeError，讓上層知道要人工檢查（而不是
    悄悄回傳一組錯誤的四邊形）。

    warning: min_warm_ratio=0.5 是根據 _is_top_face_color() 既有的門檻
    （brightness_thresh=130、warmth_thresh=20，這裡故意放寬 warmth_thresh
    到 10，避免頂面邊緣抗鋸齒/陰影像素被誤判太嚴格）拍出來的經驗值，還沒
    拿實機資料驗證過，如果正常紙箱也常常被這關擋下來，代表門檻需要調鬆。
    """
    h, w = roi_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [corners.astype(np.int32)], 255)
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise RuntimeError("偵測到的頂面四邊形沒有涵蓋任何有效像素，明顯異常，請檢查 ROI/角點座標。")

    rng = np.random.default_rng(0)
    n = min(num_samples, len(xs))
    idx = rng.choice(len(xs), size=n, replace=False)

    warm_count = 0
    for i in idx:
        if _is_top_face_color(roi_img[ys[i], xs[i]], brightness_thresh, warmth_thresh):
            warm_count += 1
    warm_ratio = warm_count / n

    if warm_ratio < min_warm_ratio:
        raise RuntimeError(
            f"偵測到的『頂面』四邊形內只有 {warm_ratio:.0%} 的抽樣像素符合紙箱"
            f"暖色特徵（門檻 {min_warm_ratio:.0%}），很可能抓到背景/桌面而不是"
            "紙箱頂面。請用 --debug 檢查 debug_box_mask.png / "
            "debug_top_face_mask.png，或確認 ROI 有沒有把桌面/雜物也框進去"
            "（常見成因：auto-ROI 候選被旁邊暖色雜物橋接、ROI 框太大）。"
        )
    return warm_ratio


def detect_seam(image, roi, debug=False):
    x0, y0, x1, y1 = roi
    h_img, w_img = image.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w_img, x1), min(h_img, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI 不合法: {roi}，畫面大小是 {w_img}x{h_img}")

    roi_img = image[y0:y1, x0:x1]
    box_mask = segment_box_mask(roi_img)
    top_mask = segment_top_face_mask(roi_img, box_mask)

    if debug:
        cv2.imwrite("debug_box_mask.png", box_mask)
        cv2.imwrite("debug_top_face_mask.png", top_mask)

    corners, top_contour = find_quad_corners(top_mask)
    corners = refine_corners_by_line_fit(top_contour, corners)
    corners = refine_corners_by_color_edges(roi_img, corners)
    _validate_top_face_quad_color(roi_img, corners)
    info = order_quad_and_find_seam(corners)

    offset = np.array([x0, y0])

    def to_full(pt):
        return (pt + offset)

    result = {
        "roi": (x0, y0, x1, y1),
        "top_face_corners": [to_full(p).tolist() for p in info["corners_ordered"]],
        "left_edge": [to_full(p).tolist() for p in info["left_edge"]],
        "right_edge": [to_full(p).tolist() for p in info["right_edge"]],
        "back_edge": [to_full(p).tolist() for p in info["back_edge"]],
        "front_edge": [to_full(p).tolist() for p in info["front_edge"]],
        "seam_line": [to_full(p).tolist() for p in info["seam_line"]],
        # back_edge / front_edge 的端點（畫面上兩條紅線的端點，共 4 個點），
        # 已依它們各自跟 left_edge / right_edge 共用的角點標好方位，
        # 詳見 order_quad_and_find_seam() 的說明
        "back_left": to_full(info["back_left"]).tolist(),
        "back_right": to_full(info["back_right"]).tolist(),
        "front_left": to_full(info["front_left"]).tolist(),
        "front_right": to_full(info["front_right"]).tolist(),
    }
    return result


def draw_auto_roi_debug(image, debug_info):
    """畫出 estimate_auto_roi() 找到的所有候選框，方便肉眼確認自動抓的
    位置合不合理——這張圖務必檢查，不要盲目信任自動 ROI 的結果。

    顏色規則：選中的候選是綠色，其餘落在位置先驗範圍內但沒被選中的候選是
    黃色，被位置先驗排除掉的候選是灰色（如果套用了 fallback，代表這次
    先驗把所有候選都排除了，灰色候選裡最終還是可能被選中，這時外框仍是
    綠色，並另外印出 FALLBACK 字樣提醒）。青色圓圈是 expected_center（位置
    先驗的預期中心點），外擴 pad 之後的最終 ROI 用藍色框起來。"""
    vis = image.copy()
    max_dist = debug_info["max_center_dist_ratio"]
    for cand in debug_info["candidates"]:
        x, y, w, h = cand["bbox"]
        is_chosen = cand is debug_info["chosen"]
        within_prior = cand["center_dist_ratio"] <= max_dist
        if is_chosen:
            color = (0, 255, 0)
        elif within_prior:
            color = (0, 255, 255)
        else:
            color = (128, 128, 128)
        thickness = 2 if is_chosen else 1
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(
            vis,
            f"area={cand['area']:.0f} ext={cand['extent']:.2f} "
            f"sol={cand['solidity']:.2f} d={cand['center_dist_ratio']:.2f}",
            (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
        )
    ecx, ecy = debug_info["expected_center"]
    cv2.drawMarker(vis, (int(ecx), int(ecy)), (255, 255, 0),
                    markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    if debug_info["used_fallback"]:
        cv2.putText(vis, "FALLBACK: no candidate within position prior",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    x0, y0, x1, y1 = debug_info["roi"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return vis


def resolve_roi(image, roi_arg, auto_roi, debug, output_dir, base_name):
    """依 --roi / --auto-roi 參數決定實際要用的 ROI。

    - auto_roi=True：呼叫 estimate_auto_roi() 自動抓大概位置，並且一定會
      存出 `<base_name>_auto_roi.png`（候選框 + 選中框 + 外擴後的最終
      ROI），不受 --debug 影響，因為這是啟發式方法，比手動 ROI 更需要
      每次都能回頭肉眼檢查有沒有抓對；--debug 額外多存顏色遮罩本身。
      這個模式下會忽略 roi_arg（若使用者同時給了 --roi 會印警告）。
    - auto_roi=False：沿用原本行為，roi_arg 為 None 時退回 DEFAULT_ROI。
    """
    if not auto_roi:
        return tuple(roi_arg) if roi_arg is not None else DEFAULT_ROI

    if roi_arg is not None:
        print("警告: 同時指定了 --auto-roi 與 --roi，--roi 會被忽略", file=sys.stderr)

    roi, debug_info = estimate_auto_roi(image)
    x0, y0, x1, y1 = roi
    chosen = debug_info["chosen"]
    print(
        f"自動抓到的 ROI: ({x0}, {y0}, {x1}, {y1})　"
        f"（選中候選框面積 {chosen['area']:.0f} px、extent {chosen['extent']:.2f}、"
        f"solidity {chosen['solidity']:.2f}、中心距離比例 {chosen['center_dist_ratio']:.2f}，"
        f"候選共 {len(debug_info['candidates'])} 個）"
    )

    os.makedirs(output_dir, exist_ok=True)
    roi_debug_path = os.path.join(output_dir, f"{base_name}_auto_roi.png")
    cv2.imwrite(roi_debug_path, draw_auto_roi_debug(image, debug_info))
    print(f"自動 ROI 候選/結果視覺化已存到: {roi_debug_path}（建議每次都檢查一下）")

    if debug:
        mask_path = os.path.join(output_dir, "debug_auto_roi_mask.png")
        cv2.imwrite(mask_path, debug_info["mask"])

    return roi


def draw_visualization(image, result):
    vis = image.copy()
    x0, y0, x1, y1 = result["roi"]
    cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), 1)

    top_face = np.array(result["top_face_corners"], dtype=int)
    cv2.polylines(vis, [top_face], True, (0, 0, 255), 2)
    for (px, py) in top_face:
        cv2.circle(vis, (px, py), 5, (0, 165, 255), -1)

    left_edge = np.array(result["left_edge"], dtype=int)
    right_edge = np.array(result["right_edge"], dtype=int)
    cv2.line(vis, tuple(left_edge[0]), tuple(left_edge[1]), (0, 255, 0), 3)
    cv2.line(vis, tuple(right_edge[0]), tuple(right_edge[1]), (0, 255, 0), 3)

    seam = np.array(result["seam_line"], dtype=int)
    cv2.line(vis, tuple(seam[0]), tuple(seam[1]), (255, 0, 255), 3)
    for (px, py) in seam:
        cv2.circle(vis, (px, py), 6, (255, 0, 255), -1)

    # 標出兩條紅線（back_edge / front_edge）各自的兩個端點，方便對照座標
    for label, key in (("BL", "back_left"), ("BR", "back_right"),
                        ("FL", "front_left"), ("FR", "front_right")):
        px, py = result[key]
        cv2.putText(vis, label, (int(px) + 8, int(py) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return vis


def main():
    parser = argparse.ArgumentParser(description="偵測紙箱頂面左右邊，估計膠帶縫隙/切割軌跡")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="輸入影像路徑")
    parser.add_argument(
        "--roi", type=int, nargs=4, default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help=f"搜尋範圍（左上/右下角座標），不指定則用預設值 {DEFAULT_ROI}",
    )
    parser.add_argument(
        "--auto-roi", action="store_true",
        help="不手動給 --roi，改用顏色特徵自動抓箱子大概位置（會忽略 --roi）。"
             "這是粗略的啟發式方法，務必檢查存出的 *_auto_roi.png 確認有沒有抓對，"
             "抓錯的話請改回手動 --roi。",
    )
    parser.add_argument("--output-dir", default=None, help="輸出資料夾，預設與輸入影像同一個資料夾")
    parser.add_argument("--debug", action="store_true", help="額外存出中間過程的遮罩圖，方便除錯")
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        print(f"錯誤：找不到影像檔案 '{args.image}'", file=sys.stderr)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"錯誤：無法讀取影像 '{args.image}'", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.image))
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.image))[0]

    try:
        roi = resolve_roi(image, args.roi, args.auto_roi, args.debug, output_dir, base_name)
        result = detect_seam(image, roi, debug=args.debug)
    except (ValueError, RuntimeError) as e:
        print(f"偵測失敗: {e}", file=sys.stderr)
        sys.exit(1)

    vis = draw_visualization(image, result)
    vis_path = os.path.join(output_dir, f"{base_name}_tape_seam.png")
    cv2.imwrite(vis_path, vis)

    json_path = os.path.join(output_dir, f"{base_name}_tape_seam.json")
    json_result = {"image": os.path.abspath(args.image), **result}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, ensure_ascii=False, indent=2)

    print("紙箱頂面 4 個角點（原圖座標，依序）：")
    for (px, py) in result["top_face_corners"]:
        print(f"  ({px:.0f}, {py:.0f})")
    print(f"左邊: {[tuple(map(int,p)) for p in result['left_edge']]}")
    print(f"右邊: {[tuple(map(int,p)) for p in result['right_edge']]}")
    (lx, ly), (rx, ry) = result["seam_line"]
    print(f"膠帶縫隙／切割軌跡（左邊中點 → 右邊中點）：({lx:.1f}, {ly:.1f}) -> ({rx:.1f}, {ry:.1f})")
    bl = result["back_left"]; br = result["back_right"]
    fl = result["front_left"]; fr = result["front_right"]
    print(
        "兩條紅線（後邊／前邊）的端點："
        f" back_left=({bl[0]:.0f}, {bl[1]:.0f})　back_right=({br[0]:.0f}, {br[1]:.0f})　"
        f" front_left=({fl[0]:.0f}, {fl[1]:.0f})　front_right=({fr[0]:.0f}, {fr[1]:.0f})"
    )
    print(f"視覺化圖片已存到: {vis_path}")
    print(f"座標 JSON 已存到: {json_path}")


if __name__ == "__main__":
    main()
