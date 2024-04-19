@echo off
chcp 65001

REM 定义Python版本和下载链接
set PYTHON_VERSION=3.8.0
set PYTHON_DOWNLOAD_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-amd64.exe

REM 下载Python安装程序
echo 下载Python安装程序...
curl -o python-%PYTHON_VERSION%-amd64.exe %PYTHON_DOWNLOAD_URL%

REM 安装Python
echo 安装Python %PYTHON_VERSION%...
python-%PYTHON_VERSION%-amd64.exe /quiet InstallAllUsers=1 PrependPath=1

REM 删除下载的安装程序
del python-%PYTHON_VERSION%-amd64.exe

pause
