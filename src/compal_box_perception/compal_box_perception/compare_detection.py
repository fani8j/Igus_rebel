"""Capture one RGB-D frame and compare baseline versus current detection overlays."""

from __future__ import annotations

import argparse
import json
import subprocess
import types
from pathlib import Path

import cv2
import message_filters
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image

import detect_tape_seam as current_detector

from .depth_utils import decode_compressed_depth, fit_top_plane_quad


class FrameCapture(Node):
    def __init__(
        self,
        rgb_topic: str,
        depth_topic: str,
        info_topic: str,
        debug_topic: str,
        slop: float,
    ):
        super().__init__("compare_box_detection_capture")
        self.camera_info = None
        self.rgbd = None
        self.debug_messages = {}
        self.messages = None
        self.info_sub = self.create_subscription(
            CameraInfo, info_topic, self._on_info, qos_profile_sensor_data
        )
        self.debug_sub = self.create_subscription(
            Image, debug_topic, self._on_debug, 10
        )
        self.rgb_sub = message_filters.Subscriber(
            self, CompressedImage, rgb_topic, qos_profile=qos_profile_sensor_data
        )
        self.depth_sub = message_filters.Subscriber(
            self, CompressedImage, depth_topic, qos_profile=qos_profile_sensor_data
        )
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=slop
        )
        self.sync.registerCallback(self._on_rgbd)

    @staticmethod
    def _stamp_key(header):
        return header.stamp.sec, header.stamp.nanosec

    def _on_info(self, message):
        self.camera_info = message

    def _on_debug(self, message):
        self.debug_messages[self._stamp_key(message.header)] = message
        if len(self.debug_messages) > 30:
            self.debug_messages.pop(next(iter(self.debug_messages)))
        self._match()

    def _on_rgbd(self, rgb, depth):
        if self.camera_info is not None:
            self.rgbd = rgb, depth, self.camera_info
            self._match()

    def _match(self):
        if self.messages is not None or self.rgbd is None:
            return
        rgb, depth, info = self.rgbd
        debug = self.debug_messages.get(self._stamp_key(rgb.header))
        if debug is not None:
            self.messages = rgb, depth, info, debug

def load_baseline(workspace: Path, revision: str):
    relative_path = "src/compal_box_perception/detect_tape_seam.py"
    source = subprocess.check_output(
        ["git", "show", f"{revision}:{relative_path}"], cwd=workspace, text=True
    )
    module = types.ModuleType("detect_tape_seam_baseline")
    module.__file__ = f"{revision}:{relative_path}"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def scaled_roi(parameters, image_shape):
    reference_width, reference_height = parameters["manual_roi_reference_size"]
    x0, y0, x1, y1 = parameters["manual_roi"]
    height, width = image_shape[:2]
    return (
        round(x0 * width / reference_width),
        round(y0 * height / reference_height),
        round(x1 * width / reference_width),
        round(y1 * height / reference_height),
    )


def expand_roi(roi, image_shape, ratio):
    x0, y0, x1, y1 = roi
    height, width = image_shape[:2]
    dx, dy = round((x1 - x0) * ratio), round((y1 - y0) * ratio)
    return max(0, x0 - dx), max(0, y0 - dy), min(width, x1 + dx), min(height, y1 + dy)


def result_from_corners(roi, corners):
    geometry = current_detector.order_quad_and_find_seam(corners)
    convert = lambda value: np.asarray(value, dtype=float).tolist()
    return {
        "roi": tuple(roi),
        "top_face_corners": convert(geometry["corners_ordered"]),
        "left_edge": convert(geometry["left_edge"]),
        "right_edge": convert(geometry["right_edge"]),
        "back_edge": convert(geometry["back_edge"]),
        "front_edge": convert(geometry["front_edge"]),
        "seam_line": convert(geometry["seam_line"]),
        "back_left": convert(geometry["back_left"]),
        "back_right": convert(geometry["back_right"]),
        "front_left": convert(geometry["front_left"]),
        "front_right": convert(geometry["front_right"]),
    }


def run_baseline(module, image):
    roi, _ = module.estimate_auto_roi(image)
    result = module.detect_seam(image, roi)
    return result, module.draw_visualization(image, result)


    manual = scaled_roi(parameters, image.shape)
    search = expand_roi(manual, image.shape, parameters["auto_roi_search_margin_ratio"])
    height, width = image.shape[:2]
    center_x = (manual[0] + manual[2]) / 2.0
    center_y = (manual[1] + manual[3]) / 2.0
    search_half_diagonal = np.hypot(
        (search[2] - search[0]) / 2.0, (search[3] - search[1]) / 2.0
    )
    roi, _ = current_detector.estimate_auto_roi(
        image,
        expected_center_ratio=(center_x / width, center_y / height),
        max_center_dist_ratio=search_half_diagonal / np.hypot(width, height),
    )
    x0, y0, x1, y1 = roi
    fit_y1 = y0 + round((y1 - y0) * parameters["plane_fit_height_ratio"])
    corners, normal, _, inlier_ratio, rms = fit_top_plane_quad(
        depth,
        info.k,
        info.d,
        scale=0.001,
        fit_roi=(x0, y0, x1, fit_y1),
        classify_roi=roi,
        sample_stride=parameters["plane_sample_stride"],
        distance_threshold_m=parameters["plane_distance_threshold_m"],
        iterations=parameters["plane_ransac_iterations"],
        min_inliers=parameters["plane_min_inliers"],
        min_inlier_ratio=parameters["plane_min_inlier_ratio"],
    )
    result = result_from_corners(roi, corners)
    if parameters["use_rgb_quad_edge_refinement"]:
        corners, _ = current_detector.refine_quad_edges_by_gradient(
            image,
            result,
            search_range=parameters["quad_edge_search_range_px"],
            offset_step=parameters["quad_edge_offset_step_px"],
            min_gradient_score=parameters["quad_edge_min_gradient_score"],
        )
        result = result_from_corners(roi, corners)
    _, ratio, score = current_detector.refine_seam_line_by_gradient(
        image,
        result,
        min_ratio=parameters["seam_ratio_min"],
        max_ratio=parameters["seam_ratio_max"],
        ratio_step=parameters["seam_ratio_step"],
        min_gradient_score=parameters["seam_min_gradient_score"],
    )
    back_left = np.asarray(result["back_left"])
    back_right = np.asarray(result["back_right"])
    front_left = np.asarray(result["front_left"])
    front_right = np.asarray(result["front_right"])
    result["seam_line"] = np.stack(
        (
            back_left + ratio * (front_left - back_left),
            back_right + ratio * (front_right - back_right),
        )
    ).tolist()
    metrics = {
        "plane_inlier_ratio": inlier_ratio,
        "plane_rms_m": rms,
        "plane_normal": normal.tolist(),
        "seam_ratio": ratio,
        "seam_gradient_score": score,
    }
    return result, current_detector.draw_visualization(image, result), metrics


def labeled(image, text, color=(255, 255, 255)):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(output, text, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    return output


def error_panel(image, title, error):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(output, title, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(output, str(error)[:100], (12, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    return output


def main(args=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("/home/farhan/igus_rebel_ros2"))
    parser.add_argument("--baseline", default="c08d3bb")
    parser.add_argument("--output", type=Path, default=Path("/tmp/compal_detector_comparison.png"))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--rgb-topic", default="/camera/color/image_raw/compressed")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw/compressedDepth")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--debug-topic", default="/box_detection/debug_image")
    options = parser.parse_args(args)

    config_path = options.workspace / "src/compal_box_perception/config/box_perception.yaml"
    parameters = yaml.safe_load(config_path.read_text())["box_perception"]["ros__parameters"]
    baseline = load_baseline(options.workspace, options.baseline)

    rclpy.init()
    capture = FrameCapture(
        options.rgb_topic,
        options.depth_topic,
        options.camera_info_topic,
        options.debug_topic,
        parameters["sync_slop_s"],
    )
    deadline = capture.get_clock().now().nanoseconds + int(options.timeout * 1e9)
    while capture.messages is None and capture.get_clock().now().nanoseconds < deadline:
        rclpy.spin_once(capture, timeout_sec=0.2)
    messages = capture.messages
    capture.destroy_node()
    rclpy.shutdown()
    if messages is None:
        raise RuntimeError(
            "timed out waiting for synchronized RGB-D and a debug image "
            "with the exact same RGB timestamp"
        )

    rgb_message, depth_message, info, debug_message = messages
    image = cv2.imdecode(
        np.frombuffer(bytes(rgb_message.data), np.uint8), cv2.IMREAD_COLOR
    )
    depth, _ = decode_compressed_depth(bytes(depth_message.data), depth_message.format)
    production_debug = CvBridge().imgmsg_to_cv2(
        debug_message, desired_encoding="bgr8"
    )
    report = {"baseline": options.baseline}

    try:
        baseline_result, baseline_overlay = run_baseline(baseline, image)
        report["baseline_result"] = baseline_result
        baseline_panel = labeled(baseline_overlay, f"BASELINE {options.baseline}")
    except Exception as error:
        report["baseline_error"] = str(error)
        baseline_panel = error_panel(image, f"BASELINE {options.baseline} FAILED", error)

    try:
        current_result, current_overlay, metrics = run_current(image, depth, info, parameters)
        report["current_result"] = current_result
        report["current_metrics"] = metrics
        current_panel = labeled(current_overlay, "CURRENT DEPTH + RGB")
    except Exception as error:
        report["current_error"] = str(error)
        current_panel = error_panel(image, "CURRENT FAILED", error)

    debug_path = options.output.with_name(
        f"{options.output.stem}_debug{options.output.suffix}"
    )
    production_panel = labeled(production_debug, "PRODUCTION DEBUG (SAME FRAME)")
    comparison = np.hstack(
        (labeled(image, "RAW RGB"), baseline_panel, current_panel, production_panel)
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(options.output), comparison)
    cv2.imwrite(str(debug_path), production_debug)
    json_path = options.output.with_suffix(".json")
    report["production_debug"] = str(debug_path)
    report["source_stamp"] = {
        "sec": rgb_message.header.stamp.sec,
        "nanosec": rgb_message.header.stamp.nanosec,
    }
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"comparison: {options.output}")
    print(f"debug: {debug_path}")
    print(f"report: {json_path}")


if __name__ == "__main__":
    main()
