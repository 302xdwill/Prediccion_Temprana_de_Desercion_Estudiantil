@echo off
title SIA-UPeU - Sistema Inteligente de Prediccion de Desercion Estudiantil (FastAPI + Keras M4)
echo ===============================================================================
echo   UNIVERSIDAD PERUANA UNION (UPeU) - INGENIERIA DE SISTEMAS
echo   SIA-UPeU: Sistema Inteligente de Prediccion Temprana de Desercion Estudiantil
echo   Autor: William David Ortiz Mamani
echo ===============================================================================
echo.
echo Iniciando servidor FastAPI y cargando Red Neuronal Profunda (mejor_modelo_desercion.keras)...
echo El navegador se abrira automaticamente en http://127.0.0.1:8000
echo.
start "" "http://127.0.0.1:8000"
"D:\Documentos\Trabajos upeu\Trabajos upeu\C8\IA\env\Scripts\python.exe" "D:\Documentos\Trabajos upeu\UPEU\Inteligencia Artificial\Articulo\webapp\app.py"
pause
