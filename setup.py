from setuptools import setup, find_packages

setup(
    name="deepkoopformer",
    version="0.1.0",
    description="Koopman-augmented transformer models for scientific time series forecasting",
    author="Ali Forootani",
    author_email="aliforootani@ieee.org",
    url="https://github.com/yourusername/deepkoopformer",  # change if available
    packages=find_packages(include=["deepkoopformer", "deepkoopformer.*"]),
    python_requires="==3.10.6",
    install_requires=[
        "numpy==1.24.4",
    "pandas==1.5.3",
    "matplotlib==3.7.1",
    "scikit-learn==1.2.2",
    "scipy==1.10.1",
    "torch==2.0.1",
    "seaborn>=0.13.2"
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "isort"],
        "docs": ["sphinx", "numpydoc"],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
