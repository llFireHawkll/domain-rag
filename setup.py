from setuptools import find_packages, setup

from domain_rag import __version__ as src_version

with open("./requirements.txt") as text_file:
    requirements = text_file.readlines()

requirements = list(map(lambda x: x.rstrip("\n"), requirements))
install_libraries = [x for x in requirements if not x.startswith("--extra-index")]


setup(
    name="domain_rag",
    version=src_version,
    description="Agentic RAG workflow purposely build for domain specific search task with context driven iterative reasoning",
    author="Sparsh Dutta",
    author_email="sparsh.dtt@gmail.com",
    packages=find_packages(include=["domain_rag", "domain_rag.*"]),
    include_package_data=True,
    install_requires=install_libraries,
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
