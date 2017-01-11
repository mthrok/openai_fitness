from __future__ import absolute_import


def _get_version_from_git():
    import os
    from subprocess import check_output

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.join(base_dir, '..')
    return check_output(['git', '-C', repo, 'describe', '--tag']).strip()


def _get_version_from_setup():
    import pkg_resources
    return pkg_resources.require('luchador')[0].version


try:
    __version__ = _get_version_from_git()
except Exception:
    __version__ = _get_version_from_setup()
