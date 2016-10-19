import os
from subprocess import check_output

base_dir = os.path.dirname(os.path.abspath(__file__))
repo = os.path.join(base_dir, '..')

__version__ = check_output(['git', '-C', repo, 'describe', '--tag']).strip()
