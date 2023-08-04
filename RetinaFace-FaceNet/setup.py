# -*- coding = utf-8 -*-
"""
# @Time : 2023/7/4 13:19
# @Author : FriK_log_ff 374591069
# @File : setup.py.py
# @Software: PyCharm
# @Function: 请输入项目功能
"""
from cx_Freeze import setup, Executable

setup(
    name = "GUI.py",
    version = "0.1",
    description = "GUI.py",
    executables = [Executable("GUI.py")]
)
