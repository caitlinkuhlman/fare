#! /usr/bin/env python

from setuptools import setup

DISTNAME = 'fare'
DESCRIPTION = 'Fare auditing diagnostics and pairwise error metrics for fair ranking.'

def readme():
    with open('README.md') as f:
        return f.read()
	
MAINTAINER = 'Caitlin Kuhlman'
MAINTAINER_EMAIL = 'cakuhlman@wpi.edu'
URL = 'https://github.com/caitlinkuhlman/fare'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/caitlinkuhlman/fare'
VERSION = '0.1'
INSTALL_REQUIRES = ['numpy', 'scipy', 'matplotlib']
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 2.7',
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=readme(),
      zip_safe=False,  # the package can run out of an .egg file
      classifiers=CLASSIFIERS,
      packages=['fare'],
	  include_package_data=True,
      install_requires=INSTALL_REQUIRES)
