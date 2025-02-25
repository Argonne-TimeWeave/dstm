"""
/*******************************************************************************
 * Copyright 2025 UChicago Argonne, LLC.
 * (c.f. AUTHORS, LICENSE)
 *
 * This file is part of the dstm project.
 * For more info, see https://github.com/anlsys/dstm
 *
 * SPDX-License-Identifier: BSD-3-Clause
 ******************************************************************************/
"""

from setuptools import setup

setup(
    name="dstm",
    version="0.1.0",
    description="DSTM Python Bindings",
    author="Swann Perarnau",
    author_email="swann@anl.gov",
    url="https://github.com/anlsys/dstm",
    license="BSD3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.10",
    ],
    packages=["dstm"],
    package_dir={"dstm": "dstm"},
)
