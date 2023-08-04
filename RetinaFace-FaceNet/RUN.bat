chcp 65001
@echo off
 
rem 检查python是否安装 
python -V >nul 2>&1
if %errorlevel% == 0 (
  echo Python已安装,版本为:
  python -V
) else (
  echo Python未安装,开始下载安装包
  curl -O https://mirrors.huaweicloud.com/python/3.8.0/python-3.8.0-amd64.exe
  echo 安装包下载完成,开始安装
  python-3.8.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
  echo Python安装完毕
)
 

rem 创建虚拟环境 
python -m venv venv
call venv\Scripts\activate.bat
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r re.txt


python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip setuptools
pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python==4.5.3.56

pip install torch-1.7.1+cu110-cp38-cp38-win_amd64.whl

pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -i https://mirrors.aliyun.com/pypi/simple/ tqdm
pip install -i https://mirrors.aliyun.com/pypi/simple/ gradio
pip install -i https://mirrors.aliyun.com/pypi/simple/ scipy
pip install -i https://mirrors.aliyun.com/pypi/simple/ matplotlib
numpy==1.21.2
pip install urllib3==1.25.11


rem 检查python是否安装 
python -V >nul 2>&1
if %errorlevel% == 0 (
  echo Python已安装,版本为:
  python -V
) else (
  echo Python未安装,开始下载安装包
  curl -O https://mirrors.huaweicloud.com/python/3.7.0/python-3.7.0-amd64.exe
  echo 安装包下载完成,开始安装
  python-3.7.0-amd64.exe /quiet InstallAllUsers=1 PrependPath=1
  echo Python安装完毕
)

python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip setuptools
pip install dlib-19.24.2-cp37-cp37m-win_amd64.whl
pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python==4.5.3.56
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-1.7.1+cu110-cp37-cp37m-win_amd64.whl
pip install -i https://mirrors.aliyun.com/pypi/simple/ tqdm==4.62.2
pip install -i https://mirrors.aliyun.com/pypi/simple/ gradio
pip install -i https://mirrors.aliyun.com/pypi/simple/ scipy==1.7.1
pip install -i https://mirrors.aliyun.com/pypi/simple/ numpy==1.21.2
pip install -i https://mirrors.aliyun.com/pypi/simple/ matplotlib
pip install -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn
pip install urllib3==1.26.9 

python -m venv venv
call venv\Scripts\activate.bat
python gui.py
python guitest.py