from __future__ import absolute_import
from setuptools import setup, find_packages
from io import open


setup(
    name=u'bps_torch',
    include_package_data=True,
    #packages=find_packages(),
    description=u'A Pytorch Inplementation of bps_torch 3D representation',
    long_description=open(u"README.md").read(),
    long_description_content_type=u"text/markdown",
    version=u'0.1',
    url=u'https://github.com/otaheri/bps_torch',
    author=u'Omid Taheri',
    author_email=u'omid.taheri@tuebingen.mpg.de',
    maintainer=u'Omid Taheri',
    maintainer_email=u'omid.taheri@tuebingen.mpg.de',
    #keywords=['pip','MANO'],
    install_requires=[
          u'numpy>=1.16.2',
          u'torch>=1.0.1.post2',
          u'torchgeometry>=0.1.2',
      ],
    packages=[u'bps_torch']
      
    )
