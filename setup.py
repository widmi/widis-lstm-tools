from setuptools import find_packages, setup

readme = open("README.md").read()
requirements = {"install": ["torch", "numpy", "matplotlib"]}
install_requires = requirements["install"]

setup(
    # Metadata
    name="widis-lstm-tools",
    version="0.1",
    author="Michael Widrich",
    author_email="widrich@ml.jku.at",
    url="https://github.com/widmi/widis-lstm-tools",
    license="GPL-3.0",
    description=(
        "Various tools for working with Long Short-Term Memory (LSTM) "
        "networks and sequences in Pytorch"
    ),
    long_description=readme,
    long_description_content_type="text/markdown",
    # Package info
    packages=find_packages(),
    zip_safe=True,
    install_requires=install_requires,
)
