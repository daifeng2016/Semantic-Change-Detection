# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by D. F. Peng on 2019/6/18
"""
import argparse
import json

import os
from bunch import Bunch

from configs.root_dir import ROOT_DIR
from utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    将配置文件转换为配置类
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)  # 配置字典

    config = Bunch(config_dict)  # 将配置字典转换为类

    return config, config_dict


def process_config(json_file):
    """
    解析Json文件
    :param json_file: 配置文件
    :return: 配置类
    """
    config, _ = get_config_from_json(json_file)


    return config


def get_train_args():
    """
    添加训练参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='path',#文件属性 path extension exclude
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser


def get_test_args():
    """
    添加测试路径
    :return: 参数
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--cfg',
        dest='config',
        metavar='C',
        default='None',
        help='add a configuration file')
    args = parser.parse_args()
    return args, parser
