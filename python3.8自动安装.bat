chcp 65001
@echo off
echo Python未安装,开始下载安装包
curl -O https://mirrors.huaweicloud.com/python/3.7.0/python-3.7.0-amd64.exe
echo 安装包下载完成,开始安装
python-3.7.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
pause
