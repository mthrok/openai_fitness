def _parse_command_line_args():
    import argparse
    parser = argparse.ArgumentParser(
        description='Train GAN.'
    )
    parser.add_argument(
        'dataset',
        description='Path to dataset file.'
    )
    return parser.parse_args()


def _main():
    args = _parse_command_line_args()


if __name__ == '__main__':
    _main()
