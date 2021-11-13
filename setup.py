from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='VeryAccurateEmulator',
    version='2.0.0',
    description='21cmVAE: A Very Accurate Emulator of the 21-cm Global Signal.',
    long_description=readme(),
    author='Christian H. Bye',
    author_email='chbye@berkeley.edu',
    url='https://github.com/christianhbye/21cmVAE',
    packages=['VeryAccurateEmulator'],
    install_requires=open('requirements.txt').read().splitlines(),
    license='MIT',
    classifiers=[
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: MIT License',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: Astronomy'
    ],
)
