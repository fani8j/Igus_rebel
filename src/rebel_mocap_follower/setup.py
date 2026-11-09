from glob import glob

from setuptools import find_packages, setup

package_name = "rebel_mocap_follower"

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
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="Safety-gated mocap joint follower for the igus ReBeL arm (MoveIt Servo JointJog).",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mocap_follower = rebel_mocap_follower.follower_node:main",
            "mocap_enable_keyboard = rebel_mocap_follower.enable_keyboard:main",
            "fake_mocap_publisher = rebel_mocap_follower.fake_mocap_publisher:main",
            "mocap_servo_scene_relay = rebel_mocap_follower.servo_scene_relay:main",
            "mocap_servo_starter = rebel_mocap_follower.servo_starter:main",
        ],
    },
)
