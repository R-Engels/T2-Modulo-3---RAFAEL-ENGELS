@echo off
setlocal EnableExtensions
chcp 65001 >nul

pushd "%~dp0" || (
    echo [ERRO] Nao foi possivel acessar a pasta do projeto.
    pause
    exit /b 1
)

set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo [ERRO] Ambiente virtual nao encontrado em "%PY%".
    echo Crie o ambiente com: python -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
    popd
    pause
    exit /b 1
)

echo ============================================================
echo   [1/3] Executando pipeline completo (src.run_pipeline)
echo ============================================================
"%PY%" -m src.run_pipeline
if errorlevel 1 (
    echo [ERRO] Falha na execucao do pipeline.
    popd
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   [2/3] Executando testes automatizados (pytest -v)
echo ============================================================
"%PY%" -m pytest -v
if errorlevel 1 (
    echo [ERRO] Falha na suite de testes.
    popd
    pause
    exit /b 1
)

if /I "%~1"=="--sem-jupyter" (
    echo.
    echo [OK] Pipeline e 17 testes concluidos com sucesso.
    popd
    exit /b 0
)

echo.
echo ============================================================
echo   [3/3] Abrindo Jupyter Notebook (Ctrl+C 2x para encerrar)
echo ============================================================
"%PY%" -m notebook "notebooks\pipeline_risco_credito.ipynb"

popd
pause
endlocal
