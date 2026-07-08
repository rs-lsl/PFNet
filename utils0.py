# -*- coding: utf-8 -*-
import datetime
import os
import shutil
import sys


def get_days_in_year(start_year, end_year):
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 1, 1)
    days = (end_date - start_date).days
    return days


def create_folder_if_not_exists(folder_path):
    os.makedirs(folder_path, exist_ok=True)


def sort_by_last_digit(s):
    # 从字符串中提取最后的数字并作为排序依据
    return int(s.split('/')[-1])


def copy_all_files(source_folder, destination_folder):
    # 创建目标文件夹（如果不存在）
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder, exist_ok=True)

    # 复制整个目录结构
    shutil.copytree(source_folder, destination_folder, dirs_exist_ok=True)


def save_command():
    # 获取当前时间
    now = datetime.datetime.now()
    # 获取命令行参数
    command = " ".join(sys.argv)
    # 指定保存命令的文件路径
    file_path = "/home/lisl/saved_command.txt"
    # 将命令写入文件
    with open(file_path, "a") as file:
        file.write("\n" + str(now) + "\n" + command)
    print(f"save command: {file_path}")