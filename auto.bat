rem 创建虚拟环境 
python -m venv venv
call venv\Scripts\activate.bat
python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --upgrade pip setuptools
pip install dlib-19.19.0-cp38-cp38-win_amd64.whl.whl
pip install -i https://mirrors.aliyun.com/pypi/simple/ opencv-python==4.5.3.56
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install -i https://mirrors.aliyun.com/pypi/simple/ scikit-learn
pip install --upgrade httpx     
pip install -i https://mirrors.aliyun.com/pypi/simple/ gradio==3.50.2
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
