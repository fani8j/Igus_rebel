from glob import glob

from setuptools import find_packages, setup

package_name = "side_cut_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    # colcon runs pytest when it finds a "test" extra.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="Passive side-cut camera probe: model-based top plane / edge / seam measurement.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "side_cut_vision_replay = side_cut_vision.replay:main",
            "side_cut_probe = side_cut_vision.probe_node:main",
            "mock_side_cut_camera = side_cut_vision.mock_camera:main",
        ],
    },
)
