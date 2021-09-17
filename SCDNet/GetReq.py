#-*- coding:utf-8 -*-
import os
import platform
import sys
import subprocess

# 找到当前目录
project_root = os.path.dirname(os.path.realpath(__file__))
# project_root = os.path.realpath(__file__)
print('当前目录' + project_root)

# 不同的系统，使用不同的命令语句

if platform.system() == 'Linux':
    command = sys.executable + ' -m pip freeze > ' + project_root + '/requirements.txt'
if platform.system() == 'Windows':
    command = '"' + sys.executable + '"' + ' -m pip freeze > "' + project_root + '\\requirements.txt"'
# if platform.system() == 'Windows':
#     command = '"' + sys.executable + '"' + ' -m pipreqs project_root'
# # 拼接生成requirements命令
print(command)
#
# 执行命令。
# os.system(command)  #路径有空格不管用
os.popen(command)  # 路径有空格，可用
# subprocess.call(command, shell=True)  #路径有空格，可用