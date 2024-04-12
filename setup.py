from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="sadsnap",
    packages=find_packages(),
    author='Adrian Rebmann',
    author_email='adrianrebmann@gmmail.com',
    version="0.0.1",
    description="long description",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=[
        "jupyter",
        "pandas",
        "numpy",
        "tqdm",
        "pm4py",
        "func_timeout",
        "transformers",
        "torch",
        "bitsandbytes",
        "accelerate",
        "langdetect",
        "scikit-learn"
    ]
)
