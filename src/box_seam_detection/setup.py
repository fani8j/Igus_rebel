from glob import glob
from setuptools import find_packages, setup

package_name = "box_seam_detection"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/config/scenes", glob("config/scenes/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=False,  # the SAM3 worker/daemon modules are run as files by the conda interpreter
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="RGB-D box top-face + tape-seam detector producing the H-cut snapshot.",
    license="MIT",
    entry_points={
        "console_scripts": [
            # ROS 2 action server (DetectBox): capture -> detect -> snapshot
            "box_seam_detector_node = box_seam_detection.ros.detect_node:main",
            # offline / CLI orchestrator (replaces seam_handoff capture_and_query_depth.py)
            "box_seam_detect = box_seam_detection.live.orchestrator:main",
            # live detector (serve mode used by the replay harness)
            "box_seam_detector = box_seam_detection.live.detector:main",
            # offline autolabel runner (replaces rgbd_captures scripts/autolabel_v2/run.py)
            "box_seam_autolabel = box_seam_detection.core.run:main",
            # hands-free capture session (saves when the box was moved and the scene is still)
            "box_seam_auto_capture = box_seam_detection.live.auto_capture:main",
        ],
    },
)
