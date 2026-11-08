@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo PlookCode Python environment not found. See README.md.
    pause
    popd
    exit /b 1
)
start "PlookCode Thai Chat" ".venv\Scripts\pythonw.exe" -m idlelib -t "PlookCode Thai Chat" -c "import sys; sys.argv=['localbrain','chat','--web']; from localbrain.cli import main; main()"
popd
endlocal
