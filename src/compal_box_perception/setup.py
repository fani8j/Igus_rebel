from glob import glob
from setuptools import find_packages, setup

package_name = "compal_box_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    py_modules=["detect_tape_seam"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/config",
            glob("config/*.yaml") + glob("config/*.rviz"),
        ),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="Live RGB-D carton seam detection in the igus ReBeL planning frame.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "box_perception_node = compal_box_perception.box_perception_node:main",
        ],
    },
)
