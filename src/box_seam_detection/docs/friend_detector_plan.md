# Adopting parts of the friend's conveyor box detector — DRAFT plan (2026-10-12)

Source: `/home/farhan/Downloads/box_detection_friend/conveyor_ws` (extracted from `~/Downloads/box detection.zip`),
detector `src/conveyor_box/rgbd_detector.py` (`detect_rgbd_boxes`, pure numpy/OpenCV, CPU only). Summary of their
method (their ChatGPT write-up + code review): calibrated floor plane + fixed belt height → per-pixel height in plane
coordinates → top-face mask = |h − H_box| ≤ 25 mm → close + 8-connected components → per component: known-size
rectangle fit in plane XY (`_fit_partition`: minAreaRect yaw candidates, 2–98 % extents, score = inside_ratio ×
exp(−outside/tol) × occupancy on a 10×6 grid), edge evidence (`_rectangle_line_evidence`: CLAHE→Sobel / p95, band
contrast 12 mm vs 12–30 mm, 24-bin continuity), touching boxes split along HoughLinesP lines (2 or 3 parallel parts),
hypothesis selection (0.8 geometry + 0.2 boundary; 0.80/0.15/0.05 for splits; ≥ 0.65). Output: top centre + yaw +
LIBRARY size (z and size not measured), 1–3 cm accuracy for suction. Their "seam" = boundary between touching boxes, no
tape / flap-gap model. YOLO not used at runtime.

Our detector (this package): SAM3 top-face mask → sub-pixel edge refine → depth cross-check (per-capture RANSAC
planes, `depth_top`) → corner fold → flap-gap seam (now the observed gap line for `large_392`) → knife cut targets
(mm accuracy). Open problems the friend's ideas could help with:
- `large_392`: the quad is ~6 mm too deep (front edge on the rounded front lip): corners and side-cut targets are off
  by up to ~6–11 mm; the measured top is 392–401 × 326–329 mm vs the 390 × 320 mm box.
- No independent check that each quad edge is supported along its whole length.
- SAM3 is the single point of failure for finding the top (GPU, ~2 GB, 0.3–1.6 s).

## Proposed steps (all validated on the 20 dual-arm captures, the 431 old-setup captures and the 3 blind sets)

| # | Step | Mode first | Promote when |
|---|---|---|---|
| F1 | **Known-size rectangle fit** of the top-plane inliers in our per-capture plane frame (port `_fit_partition`, tolerances scaled to the box: 3–5 mm, anchored on the visible edges, profile top dims), record the per-corner disagreement with the refined quad | shadow / flag | For `large_392`: replaces the front edge when the quad over-extends (e.g. quad depth − profile depth > 4 mm and the fit's edge has depth/edge support) → corners + side targets fixed; blind/real checks show improvement |
| F2 | **Per-edge continuity score** for the 4 quad edges (port `_rectangle_line_evidence`: CLAHE→Sobel normalised by the p95 in the ROI; band contrast both sides of the sub-pixel edge, bands in mm scaled to the box; 24-bin coverage) | shadow / flag (`edge_unsupported(<edge>)`) | Gate (needs_review / cut block) only if it separates the known bad quads (half-top #2, tape bands, truncated tops) from good ones on all data |
| F3 | **CPU depth-first top finder**: organized height grid from our projected cloud + per-capture plane; per profile height band (86 / 134 / 480 mm; hidden-stand scenes need a one-time stand height in the scene config) → components → F1 fit → coarse quad | fallback candidate when SAM3 finds nothing / is rejected; pre-classifier of the profile | Only after it matches SAM3 quads within a few mm on the accepted captures; never sole source for a cut without the edge/seam checks |
| F4 | **Touching / multiple boxes**: detect >1 box-sized component or a split line → reject with an operator hint (never cut) | reject reason | — |
| F5 | Gradient normalisation (CLAHE→Sobel→p95) reused in edge refine / seam scoring | experiment | only if identical-or-better on the replays |

Not adopted: their library-height z (we measure), 25 mm tolerances, fixed floor + belt height instead of per-capture
planes, lowest-box-first selection, YOLO notebook (one-class bbox, weaker than SAM3 masks), 2/3-box split as a producer.

Implementation: new module `core/known_size_fit.py` (F1, F2 helpers; pure numpy/cv2, ported with attribution) +
wiring in `pipeline.py` behind profile/scene flags; F3 in `core/autolabel/depth_first_top.py`. Old setup must stay
bit-identical until a step is promoted (431 replay + blind sets via `replay_diff`).

## Advisor review + decisions (2026-10-12)

advisor-fable (measured on the 20 captures) and advisor-astra agree: the friend's known-size rectangle fit (F1) cannot
reach mm accuracy here (top points within the plane band stop 8–29 mm short of the edges; the top is tilted/bulged, so
a centred rectangle is biased 4–10 mm sideways; profile top dims were circular). F2 (RGB gradient band score) would
not separate the known bad quads (they end on crisp lines). F5 dropped. Our RGB quad is precise but biased: front-back
325.9 ± 0.9 mm (17 captures) vs the 320 mm box.

User facts (2026-10-12): the box top is 390 × 320 mm (outer), edges slightly rounded but still sharp, the flap gap is
(more or less) centred. With the observed gap 5.5 mm behind the projective midline: front edge ~8.5 mm too far
forward (rounded lip), back edge ~2.5 mm inside.

Order of work:
1. D1 per-edge depth-support check (replaces F2): coplanar top points continuing beyond an edge = truncated quad;
   front edge: where the top surface falls away (cliff) relative to the RGB edge; back/left/right are occluding
   contours (silhouette vs floor) — record; shadow first, then cut block once validated.
2. F4 touching / multiple boxes → reject with an operator hint, only on corroborated evidence (two box-sized
   components / a depth step, not a single Hough line — the flap gap is a line too).
3. K1 gap-anchored known-size top for profiles with `top_from_gap` (large_392): front/back edges = observed gap line
   ± half the known depth (160 mm), left/right edges = RGB quad's left/right edges re-centred to the known length
   (390 mm) only if their measured length disagrees beyond tolerance (else keep RGB), in a plane re-fitted on the
   central 60 % of the quad (tilt fix); shadow (record both quads) → compare per-edge offsets (expect front ≈ −8.5 ±
   1 mm, back ≈ +2.5 mm, stable) → enable for large_392; old profiles untouched.
4. F1/F3 shadow diagnostics (known-size fit score, height-band component finder inside depth_top) — recorded only.

## Implementation result (2026-10-12, steps 1-4)

Code: `core/edge_support.py` (D1), `core/top_diag.py` (F4 + orchestration), `core/top_from_gap.py` (K1),
`core/known_size_fit.py` + `autolabel/depth_top.height_grid / height_band_components` (step 4, shadow), pipeline hook
(`pipeline.top_depth_diagnostics / top_from_gap_stage`), profile flags (`edge_support_cut_rule`, `multi_box_reject`,
`top_from_gap`; all three on for `large_392` only), detector rules (`top_edge_unsupported`, `multiple_boxes_suspect`,
rectangle gate + size check on the measured RGB quad of a K1 record), `corner_names.rename_record` permutes the new records.

- D1 (80 mm outward probe, cliff = height < -max(10 mm, 3 rms), 40 stations/edge): `unsupported` = continue share >= 0.5 AND
  median extent >= 20 mm. 20 dual-arm captures: flags #2 (front, +80 mm, 100 % of stations) and #18 (front +35, left +80 =
  the person's head); none of the 15 good ones (max 13.2 mm / 0.6 on #5 right; all others <= 5.8). 431 old captures: 16
  flagged, all already rejected (tape bands 72507/74327: back +78/+80; truncated ~100x75 tops: front +41..48; 3 more), 0 of the
  389 accepted; unflagged maximum 17.8 mm (20261001_020002_012616 front), flagged minimum 25.8 mm: old profiles stay shadow.
  Front-edge cliff in the plane (parallax-corrected): -2.4 mm (-3.8..-1.1) on the dual-arm set, +1.6 mm (p5/p95 -0.9/3.3, all
  edges facing the camera) on the old set.
- F4: evidence (other box-sized coplanar component, off-plane blob inside the top) fires on 0 of 20 dual-arm and 0 of 431 old
  captures (no real positive available; synthetic tests only): enabled for `large_392`, shadow for the rest.
- K1: see the per-edge numbers in the implementation report; outward deltas front -5.7 +- 1.0 mm (-7.4..-3.9), back -0.8 +- 0.9
  mm (-2.4..+0.8) on 17 captures, measured METRICALLY in the re-fitted plane (the expectation -8.5 / +2.5 came from the
  projective midline of the biased RGB quad; their sum, the 6 mm excess depth, agrees).
