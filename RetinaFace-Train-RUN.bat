@echo off
python -m venv venv
call venv\Scripts\activate.bat
cd  RetinaFace-Train
python gui.py
pause