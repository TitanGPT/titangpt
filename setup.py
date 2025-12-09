from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="titangpt",
    version="0.1.0",
    author="TitanGPT",
    description="A powerful GPT-based package for advanced AI capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TitanGPT/titangpt",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        # Add your project dependencies here
        # Example: "requests>=2.25.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    entry_points={
        # Add command-line entry points here if needed
        # Example: "console_scripts": ["titangpt=titangpt.cli:main"],
    },
    include_package_data=True,
    zip_safe=False,
)
