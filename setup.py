try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys
sys.path.append('VeryAccurateEmulator')

f=open('README.rst', 'r')
readme=f.read()
f.close()


setup(
    name='VeryAccurateEmulator',
    version='2.0.0',
    description='21cmVAE: A Very Accurate Emulator of the 21-cm Global Signal.',
    long_description=readme,
    author='Christian H. Bye',
    author_email='chb@berkeley.edu',
    url='https://github.com/christianhbye/21cmVAE',
    packages=['VeryAccurateEmulator'],
    include_package_data=True,
    python_requires='>=3.6',
    license='MIT',
    classifiers=[
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Astronomy'
    ],
)
