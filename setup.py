from setuptools import setup, find_packages

SHORT_DESCRIPTION = ("Protree is  a set of utilities for prototype selection in tree-based models and usage of prototypes in "
                     "drift detection.")

with open("README.md", "r", encoding="utf-8") as fp:
    LONG_DESCRIPTION = fp.read()

with open("requirements.txt", "r", encoding="utf-8") as fp:
    REQUIREMENTS = fp.read().split("\n")

with open("requirements-optional.txt", "r", encoding="utf-8") as fp:
    REQUIREMENTS_OPTIONAL = fp.read().split("\n")

setup(
    name="protree",
    version="0.0.1",
    author="jkarolczak",
    author_email="jacek.karolczak@outlook.com",
    description=SHORT_DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/jkarolczak/protree",
    packages=find_packages(where="."),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=REQUIREMENTS,
    extras_require={
        "all": REQUIREMENTS_OPTIONAL
    }
)
