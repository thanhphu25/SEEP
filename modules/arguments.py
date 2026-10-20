import argparse
import importlib
from modules.cli import add_research_args, validate


CONFIGS = ('cifar100_10task', 'cifar100_20task', 'imr_10task', 'imr_20task',
           'cub200_10task', 'cub200_20task', 'ucf101_10task', 'ucf101_20task',
           'activitynet_10task', 'activitynet_20task')


def parse_args(argv=None):
    parser = argparse.ArgumentParser('SEEP training and evaluation configs')
    sub = parser.add_subparsers(dest='subparser_name', required=True)
    for name in CONFIGS:
        child = sub.add_parser(name)
        importlib.import_module('configs.' + name).get_args_parser(child)
        add_research_args(child)
    return validate(parser.parse_args(argv))
