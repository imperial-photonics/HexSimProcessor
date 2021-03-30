import setuptools

setuptools.setup(
    name="SIM_processing",
    version="1.0.0",
    description="Image processing for SIM image data",
    url="https://github.com/ImperialCollegeLondon/HexSimProcessor",
    packages=setuptools.find_packages(),
    python_requires=">=3",
    install_requires=[
        "numpy >= 1.17.0",
    ],
)
