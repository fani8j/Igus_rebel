import os
from glob import glob

from setuptools import find_packages, setup

package_name = "dual_cut_ui"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        # web UI front end (found next to server.py in the source tree, here when installed)
        (f"share/{package_name}/ui_static", [f for f in glob(f"{package_name}/static/*") if os.path.isfile(f)]),
        # MOCK ONLY helpers (simulated detector, simulated cameras); never used on the real cell
        (f"lib/{package_name}", glob("scripts/mock_*.py")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Farhan",
    maintainer_email="farhan@example.com",
    description="Operator web UI for the dual-arm carton cutting cell.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "dual_cut_ui = dual_cut_ui.server:main",
        ],
    },
)
