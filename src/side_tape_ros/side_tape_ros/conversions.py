"""Observation <-> side_tape_interfaces/Observation, and ROS time helpers."""

from __future__ import annotations

import math

import numpy as np

from . import core_path  # noqa: F401
from precision_box_detector.side_tape.types import Observation, TrackState


def stamp_to_s(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def s_to_stamp(t: float, stamp_cls):
    msg = stamp_cls()
    sec = int(math.floor(t))
    msg.sec = sec
    msg.nanosec = int(round((t - sec) * 1e9)) % 1_000_000_000
    return msg


def observation_to_msg(obs: Observation, msg_cls, time_cls, frame_id: str = "knife_camera"):
    msg = msg_cls()
    msg.header.stamp = s_to_stamp(obs.stamp_s, time_cls)
    msg.header.frame_id = frame_id
    msg.frame_seq = int(obs.frame_seq)
    msg.state = obs.state.value
    msg.has_line = obs.line is not None
    msg.line = [float(v) for v in (obs.line if obs.line is not None else (0.0, 0.0, 0.0))]
    msg.polarity = int(obs.polarity or 0)
    msg.has_e_lat = obs.e_lat_mm is not None
    msg.e_lat_mm = float(obs.e_lat_mm) if obs.e_lat_mm is not None else float("nan")
    msg.has_e_yaw = obs.e_yaw_deg is not None
    msg.e_yaw_deg = float(obs.e_yaw_deg) if obs.e_yaw_deg is not None else float("nan")
    msg.conf = float(obs.conf)
    msg.n_inliers = int(obs.n_inliers)
    msg.rms_px = float(obs.rms_px)
    msg.proc_ms = float(obs.proc_ms)
    msg.reasons = [str(r) for r in obs.reasons]
    return msg


def msg_to_observation(msg) -> Observation:
    return Observation(
        stamp_s=stamp_to_s(msg.header.stamp),
        frame_seq=int(msg.frame_seq),
        state=TrackState(msg.state),
        line=np.asarray(msg.line, dtype=np.float64) if msg.has_line else None,
        polarity=int(msg.polarity) if msg.polarity else None,
        e_lat_mm=float(msg.e_lat_mm) if msg.has_e_lat else None,
        e_yaw_deg=float(msg.e_yaw_deg) if msg.has_e_yaw else None,
        conf=float(msg.conf),
        n_inliers=int(msg.n_inliers),
        rms_px=float(msg.rms_px),
        proc_ms=float(msg.proc_ms),
        reasons=list(msg.reasons),
    )
