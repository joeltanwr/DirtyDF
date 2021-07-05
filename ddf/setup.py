from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='dirtydf',
    version='0.0.22',
    packages=['ddf'],
    author="Joel Tan, Justin Lo, Vik Gopal",
    author_email="vik.gopal@nus.edu.sg",
    description="A package to dirty datasets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/joeltanwr/DirtyDF", 
    license="MIT",
    platforms="OS Independent",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires ='>=3.6',
    install_requires=['pandas>=1.2.1', 'statsmodels>=0.12.1', 
    'scipy>=1.6.0', 'inflection>=0.5.1', 'numpy>=1.19.5']
    #    'pandas',
    #    'statsmodels >= 0.12.1',
    #    'scipy',
    #    'inflection',
    #    'numpy; python_version >= "3.8"', ],
)
