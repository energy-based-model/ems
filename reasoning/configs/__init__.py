import argparse
import dataclasses

from .coloring_config import ColoringConfig
from .crossword_config import CrosswordConfig
from .nqueens_config import NQueensConfig
from .sat_config import SATConfig


def _str2bool(value):
    if value.lower() in ('true', 't', 'yes', 'y', '1'):
        return True
    if value.lower() in ('false', 'f', 'no', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'Expected a boolean, got {value!r}.')


def load_config(config_cls, argv=None):
    """Build a config whose defaults reproduce the paper, with optional
    command-line overrides of the form `--field value`."""
    parser = argparse.ArgumentParser(
        description=f'{config_cls.__name__}: defaults reproduce the paper\'s main configuration.'
    )

    for field in dataclasses.fields(config_cls):
        if not field.init:
            continue
        arg_type = _str2bool if field.type is bool else field.type
        parser.add_argument(
            f'--{field.name}',
            type=arg_type,
            default=None,
            help=f'(default: {field.default})',
        )

    args = parser.parse_args(argv)
    overrides = {key: value for key, value in vars(args).items() if value is not None}

    return config_cls(**overrides)
