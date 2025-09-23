# -*- coding: utf-8 -*-
# Author: Seth Z. Zhao <sethzhao506@g.ucla.edu>
# License: Academic Software License: © 2021 UCLA Mobility Lab (“Institution”).

from os.path import dirname, realpath
from setuptools import setup, find_packages, Distribution
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]


setup(
    name='OpenCOOD',
    version=__version__,
    packages=find_packages(),
    license='Academic Software License: © 2021 UCLA Mobility Lab (“Institution”)',
    author='Seth Z. Zhao',
    author_email='sethzhao506@g.ucla.edu',
    description='An open-source pytorch multi-agent system for autonomous driving '
                'cooperative perception',
    long_description=open("README.md").read(),
    install_requires=[],
)
