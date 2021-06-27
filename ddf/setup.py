from setuptools import setup

setup(
    name='ddf',
    version='0.0.1',
    packages=['ddf'],
    install_requires=[
        'pandas',
        'numpy; python_version >= "3.8"',
    ],
)
