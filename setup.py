import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tflearn", # Replace with your own username
    version="1.0.0",
    author="watson",
    author_email="watson.jiang@example.com",
    description="A small project to demo tensorflow models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/watson.jiang/tf",
    packages=setuptools.find_packages(),
    install_requires=[
                     'pandas>=1.0.3',
                     'tensorflow-gpu>=2.2.0',
                     'scikit-learn>=0.23.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
