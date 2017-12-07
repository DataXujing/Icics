import codecs
import os
import sys

try:
	from setuptools import setup
except:
	from distutils.core import setup



def read(fname):
	return codecs.open(os.path.join(os.path.dirname(__file__), fname)).read()

long_des = read("README.rst")
    
platforms = ['linux/Windows']
classifiers = [
    'Development Status :: 3 - Alpha',
    'Topic :: Text Processing',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
]

install_requires = [
    'numpy>=1.13.0',
    'pandas>=0.19.0'
    'sklearn<=0.20.0'
]

    
setup(name='Icics',
      version='0.2.0',
      description='A package that used in Inter Credit case assign.',
      long_description=long_des,
      py_modules=['icics'],
      author = "DataXujing",  
      author_email = "xujing@inter-credit.net" ,
      url = "https://dataxujing.github.io" ,
      license="Apache License, Version 2.0",
      platforms=platforms,
      classifiers=classifiers,
      install_requires=install_requires
      
      )   
      
      

  