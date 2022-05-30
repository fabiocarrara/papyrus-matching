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
    # classifiers=[
    #     "License :: OSI Approved :: MIT License",
    #     "Programming Language :: Python :: 3",
    #     "Programming Language :: Python :: 3.7",
    #     "Programming Language :: Python :: 3.8",
    # ],
    packages=["papyrus_matching"],
    include_package_data=True,
    
    # Only do this last part if your package contains third-party python packages
    install_requires=["torch==1.11.0, torchvision==0.12.0"],
)