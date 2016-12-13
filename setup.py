from __future__ import print_function

import os
import subprocess
import setuptools

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALE_ROM_DIR = os.path.join('env', 'ale', 'rom')


def download(url):
    """Downlaod data to memory"""
    import urllib2
    import StringIO

    print('Downloading ROM file')
    remote_f = urllib2.urlopen(url)
    local_f = StringIO.StringIO()
    local_f.write(remote_f.read())
    local_f.seek(0)
    return local_f


def extract(content, output_dir):
    """Extract zip file to directory"""
    import zipfile
    zf = zipfile.ZipFile(content, 'r')
    for name in zf.namelist():
        _, name_ = os.path.split(name)
        if not name_:
            continue

        output_path = os.path.join(output_dir, name_)
        print('Extracting:', output_path)

        with open(output_path, 'w') as f:
            f.write(zf.open(name).read())


class DownloadALECommand(setuptools.Command):
    """Download ALE ROM files"""
    description = 'Download ALE roms'
    user_options = [
        ('url=', None, 'URL to zipped ROMs'),
        ('output-dir=', None, 'Output file'),
    ]

    def initialize_options(self):
        """Default options"""
        self.url = (
            'https://groups.google.com/group/'
            'arcade-learning-environment/attach/10bc8e55829890/'
            'supported_roms.zip?part=0.1&authuser=0&view=1'
        )
        self.output_dir = os.path.join(BASE_DIR, 'luchador', ALE_ROM_DIR)

    def finalize_options(self):
        # Ensure the target directory exists
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def run(self):
        content = download(self.url)
        extract(content, self.output_dir)


def get_git_revision(no_commit=False):
    cmd = ['git', '-C', BASE_DIR, 'describe', '--tag']
    if no_commit:
        cmd.append('--abbrev=0')
    return subprocess.check_output(cmd).strip()


def write_version_file():
    filepath = os.path.join(BASE_DIR, 'luchador', 'version.py')

    with open(filepath, 'w') as f:
        f.write("__version__ = '{}'\n".format(get_git_revision()))


def do_setup():
    setuptools.setup(
        name='luchador',
        version=get_git_revision(no_commit=True),
        cmdclass={
            # TODO: Add custom install/build command which run `download_ale`
            'download_ale': DownloadALECommand,
        },
        packages=[
            'luchador',
            'luchador.nn',
            'luchador.nn.core',
            'luchador.nn.core.base',
            'luchador.nn.core.theano',
            'luchador.nn.core.tensorflow',
            'luchador.agent',
            'luchador.env',
            'luchador.env.ale',
            'luchador.env.cart_pole',
            'luchador.env.flappy_bird',
            'luchador.command'
        ],
        entry_points={
            'console_scripts': [
                'luchador = luchador.command.main:entry_point',
            ]
        },
        test_suite='tests.unit',
        install_requires=[
            'Pillow',  # For scipy.misc.imresize
            'h5py',
            'pyyaml',
            'pygame',
            'pyglet',
            'flask',
            'cherrypy',
        ],
        package_data={
            'luchador': [
                'nn/data/*.yml',
                '{}/*.bin'.format(ALE_ROM_DIR),
            ],
        },
    )


if __name__ == '__main__':
    write_version_file()
    do_setup()
