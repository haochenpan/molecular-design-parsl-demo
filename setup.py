# Only `chemfunctions` is installed as a package because it is the only module
# that Parsl workers need on remote compute nodes.  Parsl serializes (pickles)
# the functions defined in chemfunctions and ships them to workers; when workers
# deserialize them, Python must be able to `import chemfunctions`.
#
# The other modules (main, thinkers, configs) run exclusively on the login /
# head node where they are already importable from the working directory.
from setuptools import setup

setup(
    name='chemfunctions',
    version='0.0.1',
    py_modules=['chemfunctions'],
    description='Utilities for using quantum chemistry to design electrolyte molecules'
)
