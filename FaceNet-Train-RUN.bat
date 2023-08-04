python -m venv venv
call venv\Scripts\activate.bat
cd  FaceNet-Train
python txt_annotation.py
python gui.py
pause