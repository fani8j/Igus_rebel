"""End-to-end fixtures (node + replayed saved captures). Skipped unless BOX_SEAM_E2E=1 (needs GPU/SAM3 + ROS Humble).

    source /opt/ros/humble/setup.bash && source install/setup.bash
    BOX_SEAM_E2E=1 python3 -m pytest src/box_seam_detection/test/e2e -v -s
"""

import contextlib
import os
import time
from pathlib import Path

import pytest

from . import e2e_lib as L

CAPTURE_IDS = {
    "accepted_a": "frame_000001_20260928_040422_447556",
    "accepted_b": "frame_000001_20260917_070124_398815",
    "r2_highyaw": "frame_000001_20261002_074415_767296",
    "rejected_rect": "frame_000001_20261001_020002_012616",
    "blocked_top_face": "frame_000001_20261002_074138_759282",
    "blocked_soft_range": "frame_000001_20261001_073940_547989",
}


def pytest_collection_modifyitems(config, items):
    if os.environ.get("BOX_SEAM_E2E") == "1":
        return
    skip = pytest.mark.skip(reason="set BOX_SEAM_E2E=1 to run the node end-to-end check (GPU/SAM3, ROS Humble)")
    for item in items:
        if "/test/e2e/" in str(item.fspath).replace("\\", "/"):
            item.add_marker(skip)


@pytest.fixture(scope="session")
def scratch(tmp_path_factory) -> Path:
    root = os.environ.get("BOX_SEAM_E2E_SCRATCH")
    path = Path(root) if root else tmp_path_factory.mktemp("box_seam_e2e")
    path.mkdir(parents=True, exist_ok=True)
    assert path.resolve() != L.REAL_SNAPSHOT.parent.resolve(), "scratch must not be the real outputs dir"
    return path


@pytest.fixture(scope="session")
def stack(scratch):
    """The node (own session, isolated domain, snapshot/output in scratch) + an rclpy client."""
    for cid in CAPTURE_IDS.values():
        if not (L.CAPTURES / f"{cid}.json").is_file():
            pytest.skip(f"saved capture {cid} not available under {L.CAPTURES}")
    node = L.start_node(scratch, ["capture_timeout_s:=4.0"])
    client = L.E2EClient()
    try:
        assert client.wait_ready(120), f"node/daemon not ready, see {scratch / 'node.log'}"
        yield {"node": node, "client": client, "scratch": scratch, "snap": scratch / "snap.json"}
    finally:
        client.close()
        node.stop()


@pytest.fixture
def replay(stack):
    """context manager: replay one saved capture on the fake camera topics while the block runs."""
    @contextlib.contextmanager
    def _replay(capture_id):
        cam = L.start_camera(stack["scratch"], capture_id)
        time.sleep(4.0)  # load PCD + first frames
        try:
            yield cam
        finally:
            cam.stop()
            time.sleep(0.5)
    return _replay
