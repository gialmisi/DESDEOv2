import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh.readlines()]

with open("requirements-dev.txt", "r") as fh:
    requirements_dev = [line.strip() for line in fh.readlines()]

setuptools.setup(
    name="DESDEOv2",
    version="0.0.1",
    author="Giovanni Misitano",
    author_email="giovanni.a.misitano@jyu.fi",
    description=(
        "DESDEOv2 is a free and open source Python-based framework for "
        "developing and experimenting with interactive multiobjective "
        "optimization. This is a rewrite of the original DESDEO framework."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gialmisi/DESDEOv2",
    packages=setuptools.find_packages(exclude=["tests"]),
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    tests_require=["pytest"],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Linux",
    ],
)
