from setuptools import setup, find_packages

setup(
    name="acousticguard",
    version="1.0.0",
    description="Non-contact acoustic anomaly detection for industrial machines",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="AcousticGuard Contributors",
    url="https://github.com/yourusername/AcousticGuard",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="predictive maintenance acoustic anomaly detection machine learning pytorch",
    entry_points={
        "console_scripts": [
            "acousticguard=inference:main",
        ],
    },
)
