"""The optional accelerator uses the same header as the native CLI."""
import os
import shlex
import shutil
import sysconfig

from setuptools import Extension, setup

# A relocated CPython may advertise a compiler absent on the end user's host.
# Keep explicitly supplied toolchains and the platform's normal build defaults.
if os.name == "posix" and "CC" not in os.environ:
    configured = shlex.split(sysconfig.get_config_var("CC") or "")
    if configured and not shutil.which(configured[0]) and shutil.which("cc"):
        os.environ["CC"] = "cc"

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
