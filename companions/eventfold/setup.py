"""The optional accelerator uses the same header as the native CLI."""
from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "eventfold._native",
            sources=["src/eventfold/_native.c"],
            include_dirs=["."],
            depends=["eventfold.h"],
            optional=True,
        ),
    ],
)
