from setuptools import setup
from setuptools import find_packages
import re
from l2l.version import FULL_VERSION

"""
This file installs the l2l package.
Note that it does not perform any installation of the documentation. For this, follow the specified procedure in the
 README. For updating the version, update MAJOR_VERSION and FULL_VERSION in l2l/version.py
"""


def get_requirements(filename):
    """
    Helper function to read the list of requirements from a file
    """
    dependency_links = []
    with open(filename) as requirements_file:
        requirements = requirements_file.read().strip('\n').splitlines()
    return requirements, dependency_links


requirements, dependency_links = get_requirements('requirements.txt')
setup(
    name="L2L",
    version=FULL_VERSION,
    packages=find_packages("."),
    author="Anand Subramoney, Arjun Rao",
    author_email="anand@igi.tugraz.at, arjun@igi.tugraz.at",
    description="This module provides the infrastructure create optimizers and "
                "optimizees in order to implement learning-to-learn",
    setup_requires=['Cython', 'numpy'],
    install_requires=requirements,
    provides=['l2l'],
    dependency_links=dependency_links,
)

