@echo off
rem Abre StreamClipper: inicia el servidor local (solo 127.0.0.1) y abre el navegador.
rem Uso: doble clic. Opciones: --no-open (no abrir el navegador)  --port 8781
setlocal
cd /d "%~dp0"
title StreamClipper
set "PY=%~dp0backend\.venv\Scripts\python.exe"

if not exist "%PY%" (
  echo No se encontro el entorno de Python. Se ejecuta la instalacion primero...
  call "%~dp0instalar.bat"
  if errorlevel 1 exit /b 1
)

echo Iniciando StreamClipper...
"%PY%" -m streamclipper app %*
if errorlevel 1 (
  echo.
  echo StreamClipper se cerro con un error. Revisa el mensaje de arriba.
  echo Para diagnosticar: backend\.venv\Scripts\python.exe -m streamclipper doctor
  pause
)
endlocal
