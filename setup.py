"""Install Luchador"""
from __future__ import print_function

import os
import subprocess
import setuptools
import setuptools.command.install
import setuptools.command.develop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _get_git_revision(no_commit=False):
    cmd = ['git', '-C', BASE_DIR, 'describe', '--tag']
    if no_commit:
        cmd.append('--abbrev=0')
    return subprocess.check_output(cmd).strip()


def _get_version():
    try:
        return _get_git_revision(no_commit=True)
    except Exception as error:  # pylint: disable=broad-except
        print(error)
        return 'v0.10.1'


def _setup():
    setuptools.setup(
        name='luchador',
        version=_get_version(),
        packages=setuptools.find_packages(),
        test_suite='tests.unit',
        install_requires=[
            'h5py',
            'ruamel.yaml',
        ],
        package_data={
            'luchador': [
                'nn/data/*.yml',
            ],
        },
    )


if __name__ == '__main__':
    _setup()
