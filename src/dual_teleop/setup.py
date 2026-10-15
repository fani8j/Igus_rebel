from glob import glob

from setuptools import find_packages, setup

package_name = "dual_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="fani8j",
    maintainer_email="hucenrotiacompal@gmail.com",
    description="Per-arm MoveIt Servo teleop (arm selector, interlocks) for the dual-arm cell.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "teleop_selector = dual_teleop.selector_node:main",
            "teleop_console = dual_teleop.console_node:main",
        ],
    },
)
