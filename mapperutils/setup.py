from setuptools import setup, find_packages
import sys

setup(name='mapperutils',
      author='Emerson G. Escolar',
      version='0.1.0',
      description='Mapper Utilities',
      packages=find_packages(exclude=["*.tests"]),
      install_requires=['numpy', 'matplotlib', 'kmapper']
)
