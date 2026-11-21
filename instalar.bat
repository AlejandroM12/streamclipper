@echo off
rem Instala StreamClipper en esta carpeta: entorno de Python aislado, dependencias y FFmpeg portable.
rem Se puede ejecutar varias veces: no repite lo que ya esta hecho. No instala nada global ni modifica el PATH.
setlocal
cd /d "%~dp0"
title Instalacion de StreamClipper
echo === Instalacion de StreamClipper ===

where python >nul 2>&1
if errorlevel 1 (
  echo.
  echo No se encontro Python. Instala Python 3.11 o superior desde https://www.python.org/downloads/
  echo y marca la casilla "Add python.exe to PATH". Luego ejecuta este archivo otra vez.
  pause
  exit /b 1
)

if not exist "backend\.venv\Scripts\python.exe" (
  echo Creando el entorno de Python en backend\.venv ...
  python -m venv backend\.venv
  if errorlevel 1 (
    echo No se pudo crear el entorno virtual.
    pause
    exit /b 1
  )
)

echo Instalando dependencias (la primera vez tarda unos minutos)...
"backend\.venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
"backend\.venv\Scripts\python.exe" -m pip install --quiet -e "backend"
if errorlevel 1 (
  echo No se pudieron instalar las dependencias. Revisa tu conexion a internet.
  pause
  exit /b 1
)

if not exist "tools\ffmpeg\ffmpeg.exe" (
  echo Descargando FFmpeg portable en tools\ffmpeg ...
  powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\instalar_ffmpeg.ps1"
  if errorlevel 1 (
    echo No se pudo descargar FFmpeg. Descargalo a mano desde https://www.gyan.dev/ffmpeg/builds/
    echo ^(ffmpeg-release-essentials.zip^) y copia ffmpeg.exe y ffprobe.exe a tools\ffmpeg\
    pause
    exit /b 1
  )
)

echo.
"backend\.venv\Scripts\python.exe" -m streamclipper doctor
echo.
echo Listo. Abri StreamClipper con iniciar_streamclipper.bat
endlocal
