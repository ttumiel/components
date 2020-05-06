import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "components",
    version = "0.0.1",
    author = "Thomas Tumiel",
    description = ("Components for ML development."),
    license = "MIT",
    packages=['components'],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
    ],
)
