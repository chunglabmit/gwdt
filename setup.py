import numpy as np
import os
from setuptools import setup, Extension

version = "0.1.0"

with open("./README.md") as fd:
    long_description = fd.read()

gwdt_extension = Extension(
    name="gwdt.gwdt_impl",
    sources=[os.path.join("gwdt", "gwdt_impl.pyx")],
    include_dirs=[np.get_include()],
    language="c++"
)

setup(
    name="gwdt",
    version=version,
    description=
    "Grey-weighted distance transform",
    long_description=long_description,
    install_requires=[
        "numpy"
    ],
    setup_requires=[
        "Cython",
        "numpy"
    ],
    author="Kwanghun Chung Lab",
    packages=["gwdt"],
    url="https://github.com/chunglabmit/gwdt",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Programming Language :: Python :: 3.5',
    ],
    ext_modules=[gwdt_extension],
    zip_safe=False
)
