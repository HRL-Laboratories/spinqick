from setuptools import find_packages, setup

with open("VERSION.txt", "r") as f:
    __version__ = f.read().strip()

# Specify UTF-8 to guard against systems that default to an ASCII locale.
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="spinqick",
    version=__version__,
    packages=find_packages("src"),
    package_dir={"": "src"},
    # Uncomment next line if you need to include data files in installed packages
    # include_package_data=True # Recommend using MANIFEST.in to specify package files
    long_description=long_description,
    long_description_content_type="text/markdown",
    # pip install not supported, only conda
    # dependencies should be specified in dev_environment.yml and conda/meta.yaml
    install_requires=[
        "lmfit",
        "matplotlib",
        "netcdf4",
        "numpy<=1.26.0",
        "pydantic",
        "pydantic-settings",
        "scipy",
        "Pyro4",
        "scikit-learn",
    ],
    python_requires=">=3.10",
)
