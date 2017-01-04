"""Convert Python script into IPython Notebook"""
import argparse
from nbformat import v3, v4


def _parse_command_line_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('input', help='Input Python file')
    ap.add_argument('--output', help='Output IPython Notebook name.')
    return ap.parse_args()


def _load(filename):
    with open(filename, 'r') as file_:
        return file_.read()


def _save(data, filename):
    with open(filename, 'w') as file_:
        file_.write(data)


def _main():
    args = _parse_command_line_args()
    data = _load(args.input)
    data = (
        '{}\n'
        '# <markdowncell>\n'
        '\n'
        '# If you can read this, reads_py() is no longer broken!\n'
    ).format(data)
    data = v4.writes(v4.upgrade(v3.reads_py(data)))
    _save(data, args.output or args.input.replace('.py', '.ipynb'))


if __name__ == '__main__':
    _main()
