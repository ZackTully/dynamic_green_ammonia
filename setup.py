#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Zachary Tully",
    author_email='ztully@nrel.gov',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Analysis scripts for dynamic green ammonia wrapped around NREL HOPP software",
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dynamic_green_ammonia',
    name='dynamic_green_ammonia',
    packages=find_packages(include=['dynamic_green_ammonia', 'dynamic_green_ammonia.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/ZackTully/dynamic_green_ammonia',
    version='0.1.0',
    zip_safe=False,
)
