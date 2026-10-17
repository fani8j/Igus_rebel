"""Test isolation for the SAM3 daemon / BLAS env.

In-process orchestrator tests set os.environ directly
(AUTOLABEL_V2_SAM3_DAEMON, OPENBLAS_NUM_THREADS, AUTOLABEL_V2_BLAS_PINNED);
restore them after every test, and point every test at a private daemon dir
and singleton lock name so no test can ever touch the real daemon.
"""
import os
import tempfile
import uuid

import pytest

_RESTORE = ("AUTOLABEL_V2_SAM3_DAEMON", "OPENBLAS_NUM_THREADS",
            "AUTOLABEL_V2_BLAS_PINNED", "AUTOLABEL_V2_SAM3_DAEMON_DIR",
            "AUTOLABEL_V2_SAM3_DAEMON_LOCK_NAME", "BOX_SEAM_CAPTURE_STAGING")


@pytest.fixture(autouse=True)
def _isolate_sam3_daemon_env():
    saved = {k: os.environ.get(k) for k in _RESTORE}
    # short path: AF_UNIX socket paths are limited to ~100 bytes
    d = tempfile.mkdtemp(prefix="s3i")
    os.environ["AUTOLABEL_V2_SAM3_DAEMON_DIR"] = d
    os.environ["BOX_SEAM_CAPTURE_STAGING"] = "off"     # tests never stage into the real /dev/shm
    os.environ["AUTOLABEL_V2_SAM3_DAEMON_LOCK_NAME"] = \
        f"test-autolabel_v2_sam3-{uuid.uuid4().hex}"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import shutil
        shutil.rmtree(d, ignore_errors=True)
