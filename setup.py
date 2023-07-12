from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fp:
    readme = fp.read()

setup(
    name="dlprog",
    description="A progress bar that aggregates the values of each iteration.",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=["dlprog"],
    package_dir={"dlprog": "dlprog"},
    python_requires=">=3",
    url="https://github.com/misya11p/dlprog",
    project_urls={
        "Repository": "https://github.com/misya11p/dlprog",
    },
    author="misya11p",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="iterator progress bar aggregate deep-learning machine-learning",
    license="MIT",
)
