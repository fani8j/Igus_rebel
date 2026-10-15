import os
from glob import glob

from setuptools import find_packages, setup

package_name = "rebel_dual_arm_calibration"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        # web UI front end (found next to ui/server.py in the source tree, here when installed)
        (f"share/{package_name}/ui_static", [f for f in glob(f"{package_name}/ui/static/*") if os.path.isfile(f)]),
        # MOCK ONLY helpers (simulated scene camera, mock pose driver, truth comparison); never used on a real robot
        (f"lib/{package_name}", glob("scripts/mock_*.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="Dual-arm eye-to-hand calibration: scene camera and arm_right base pose in the arm_left_base_link frame.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "recorder_node = rebel_dual_arm_calibration.recorder_node:main",
            "solve = rebel_dual_arm_calibration.solve_cli:main",
            "generate_boards = rebel_dual_arm_calibration.targets:board_cli",
            "generate_tiles = rebel_dual_arm_calibration.targets:tiles_cli",
            "calibrate_intrinsics = rebel_dual_arm_calibration.intrinsics:main",
            "check_points = rebel_dual_arm_calibration.check_points_cli:main",
            "calibration_ui = rebel_dual_arm_calibration.ui.server:main",
        ],
    },
)
