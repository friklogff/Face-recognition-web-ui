@echo off
python -m venv venv
call venv\Scripts\activate.bat
cd  RetinaFace-FaceNet
python gui.py
pause
