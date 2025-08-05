#!/usr/bin/env python3
"""
Setup configuration for Chrono-MemBench.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements (fallback - conda is preferred)
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() 
            and not line.startswith('#') 
            and not line.startswith('-c')
            and not line.startswith('-f')
        ]

setup(
    name="algoverse-chrono-membench",
    version="0.1.0",
    description="A Two-Phase Framework for Tracking and Steering Memory Features in Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Chrono-MemBench Team",
    author_email="",
    url="https://github.com/StanchPillow55/Algoverse---Chrono-MemBench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    namespace_packages=['algoverse'],
    python_requires=">=3.12",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "isort>=5.12.0",
            "yapf>=0.32.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "jupyter": [
            "jupyter",
            "notebook", 
            "jupyterlab",
            "ipywidgets",
        ],
        "gpu": [
            "bitsandbytes>=0.42.0",
            "flash-attn>=2.0.0",
            "triton>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "chrono-bootstrap=scripts.bootstrap_env:main",
            "chrono-train=src.chrono.chrono_train:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="transformer, attention, memory, sparse-autoencoder, interpretability",
    project_urls={
        "Bug Reports": "https://github.com/StanchPillow55/Algoverse---Chrono-MemBench/issues",
        "Source": "https://github.com/StanchPillow55/Algoverse---Chrono-MemBench",
    },
)
