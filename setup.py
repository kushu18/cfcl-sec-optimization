"""Setup and packaging configuration for CFCL SEC Optimization Engine."""

from pathlib import Path
from setuptools import find_packages, setup

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="cfcl-sec-optimization",
    version="2.0.0",
    author="Kushagra Bhatt",
    author_email="kushagrabhatt2004@gmail.com",
    description="Real-Time Specific Energy Consumption (SEC) Optimization Engine for CFCL Gadepan Complex",
    long_description=open("README.md", "r", encoding="utf-8").read() if Path("README.md").exists() else "",
    long_description_content_type="text/markdown",
    url="https://github.com/kushu18/cfcl-sec-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Manufacturing",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cfcl-api=src.api.main:main",
        ],
    },
)
