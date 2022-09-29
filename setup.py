import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    version="0.0.0",
    name="ania",
    author="Fynn Firouz Faber",
    author_email="faberf@ethz.ch",
    description="Abstract Noise Injection Algorithm",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/FYN/ania",
    packages=["ania"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
    ],

)
