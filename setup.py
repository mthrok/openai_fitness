"""Install Luchador"""
import os
import setuptools
import setuptools.command.install
import setuptools.command.develop

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _setup():
    setuptools.setup(
        name='luchador',
        version='0.12.0',
        packages=setuptools.find_packages(),
        test_suite='tests.unit',
        install_requires=[
            'six',
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
