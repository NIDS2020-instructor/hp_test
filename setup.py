from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    long_description = readme_file.read()

setup(
    name = "happiness_project",
    version = "0.0.1",
    author = "Michael Dayan",
    author_email = "michael.dayan@fcbg.ch",
    description = "Predict country happiness from World Happiness Report data",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/NIDS2020-instructor/happiness_project",
    packages = find_packages(),
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.7',
    install_requires=["pandas", "xlrd", "scikit-learn"]
)