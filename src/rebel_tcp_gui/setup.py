from glob import glob

from setuptools import find_packages, setup

package_name = "rebel_tcp_gui"

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
    # colcon runs pytest when it finds a "test" extra; extras_require works with
    # both the system and newer setuptools (the latter drops tests_require).
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="Interactive tkinter pivot-calibration GUI for the ReBeL knife TCP.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "tcp_gui = rebel_tcp_gui.tcp_gui_node:main",
        ],
    },
)
