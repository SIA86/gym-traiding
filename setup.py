from setuptools import setup, find_packages

setup(
    name="gym_examples",
    version="0.0.1",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.26.0", 
        "pygame>=2.1.0"
    ],
)