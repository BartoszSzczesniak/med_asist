from setuptools import setup, find_packages

with open("readme.md", "r") as file:
    long_descr = file.read()

print(long_descr)

setup(
    name="med_assist",
    version="0.3.0",
    description="Internal package for medical assistant project.",
    long_description=long_descr,
    long_description_content_type="text/markdown",
    author="Bartosz Szcześniak",
    author_email="szczesniak.bartosz@gmail.com",
    url="https://github.com/BartoszSzczesniak/med_assist",
    packages=['med_assist', 'med_assist.components'],
    package_dir={"med_assist": "med_assist"},
    include_package_data=True,
    package_data={"med_assist": ["data/*",]},
    )