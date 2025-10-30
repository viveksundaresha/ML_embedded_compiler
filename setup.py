"""
Setup script for ML Embedded Compiler
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="ml-embedded-compiler",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Automated ML Model Compiler for Embedded ARM Systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ml-embedded-compiler",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Embedded Systems",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "tensorflow>=2.6.0",
        "tensorflow-model-optimization>=0.7.0",
        "numpy>=1.19.0",
    ],
    entry_points={
        "console_scripts": [
            "ml-embedded-compiler=cli:main",
        ],
    },
    include_package_data=True,
    keywords="ml embedded compiler quantization pruning arm optimization",
)
