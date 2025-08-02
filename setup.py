#!/usr/bin/env python3
"""
Setup script for TurboDedup
"""

from setuptools import setup, find_packages
import os
import sys

# Ensure Python 3.8+
if sys.version_info < (3, 8):
    sys.exit("Python 3.8 or higher is required")

# Read version from __init__.py
def get_version():
    with open("turbodedup/__init__.py", "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("__version__"):
                return line.split('"')[1]
    raise RuntimeError("Version not found")

# Read long description from README
def get_long_description():
    with open("README.md", "r", encoding="utf-8") as f:
        return f.read()

# Core requirements (minimal dependencies)
core_requirements = []

# Enhanced features requirements (optional)
gpu_requirements = [
    "pycuda>=2022.2.2",
    "pyopencl>=2023.1.4",
    "numpy>=1.24.0",
]

similarity_requirements = [
    "Pillow>=10.0.0",
    "ImageHash>=4.3.1", 
    "librosa>=0.10.0",
    "numba>=0.56.0",
    "soundfile>=0.12.1",
    "resampy>=0.4.0",
    "python-Levenshtein>=0.21.0",
    "PyPDF2>=3.0.0",
    "python-docx>=0.8.11",
    "chardet>=5.0.0",
    "scipy>=1.10.0",
]

dev_requirements = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "black>=23.7.0",
    "flake8>=6.0.0",
    "mypy>=1.5.0",
]

all_requirements = gpu_requirements + similarity_requirements

setup(
    name="turbodedup",
    version=get_version(),
    author="TurboDedup Team",
    author_email="info@turbodedup.dev",
    description="High-Performance File Deduplication Scanner with GPU Acceleration & Smart Caching",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/arjaygg/TurboDedup",
    project_urls={
        "Bug Reports": "https://github.com/arjaygg/TurboDedup/issues",
        "Source": "https://github.com/arjaygg/TurboDedup",
        "Documentation": "https://github.com/arjaygg/TurboDedup/blob/main/README.md",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Archiving",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
        "Environment :: Console",
    ],
    python_requires=">=3.8",
    install_requires=core_requirements,
    extras_require={
        "gpu": gpu_requirements,
        "similarity": similarity_requirements,
        "all": all_requirements,
        "dev": dev_requirements,
    },
    entry_points={
        "console_scripts": [
            "turbodedup=turbodedup.cli.main:main",
            "turbo-dedup=turbodedup.cli.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="deduplication, duplicate-files, file-management, gpu-acceleration, performance, similarity-detection",
)