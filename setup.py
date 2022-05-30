from setuptools import setup

def readme():
    with open('README.md') as f:
        README = f.read()
    return README

setup(
    name="papyrus-matching",
    version="0.1.0",
    description="A Python package for matching papyrus fragments with Deep Learning.",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/fabiocarrara/papyrus-matching",
    author="Fabio Carrara",
    author_email="fabio.carrara@isti.cnr.it",
    license="MIT",
    packages=["papyrus_matching"],
    install_requires=["torch==1.11.0", "torchvision==0.12.0"],
)