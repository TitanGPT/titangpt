from setuptools import setup, find_packages

setup(
    name="titangpt",
    version="0.1.0",
    author="TitanGPT",
    author_email="info@titangpt.ru",
    description="Official Python client for TitanGPT API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TitanGPT/titangpt",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python ::  3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System ::  OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.28.0",
        "aiohttp>=3.8.0",
    ],
)
