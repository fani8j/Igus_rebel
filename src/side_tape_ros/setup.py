from glob import glob

from setuptools import setup

package_name = "side_tape_ros"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="fani8j",
    maintainer_email="hucenrotiacompal@gmail.com",
    description="Knife-camera side-tape tracking + MoveIt Servo lateral correction",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = side_tape_ros.perception_node:main",
            "controller_node = side_tape_ros.controller_node:main",
            "sequencer = side_tape_ros.sequencer:main",
            "calib_tip_mask = side_tape_ros.calib_tip_mask:main",
            "calib_jacobian = side_tape_ros.calib_jacobian:main",
        ],
    },
)
