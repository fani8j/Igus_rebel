from glob import glob

from setuptools import find_packages, setup

package_name = "rebel_gripper_io"

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
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="SMC JXC61 gripper handshake over the igus ReBeL digital I/O.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "jxc_gripper_node = rebel_gripper_io.jxc_gripper_node:main",
        ],
    },
)
