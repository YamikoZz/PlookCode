@echo off
rem Chat with Qwen2.5-Coder-7B-Instruct (Apache-2.0) through llama.cpp. Separate helper: NOT part of the PlookCode model.
rem Files (not in git): workspace\llamacpp\ (llama.cpp b12207 CUDA 12.4) and workspace\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf
chcp 65001 >nul
pushd "%~dp0"
set MODEL=workspace\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf
if not exist "%MODEL%" (
  echo Model file missing: %MODEL%
  echo Download qwen2.5-coder-7b-instruct-q4_k_m.gguf from huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF
  pause
  popd
  exit /b
)
echo Loading Qwen2.5-Coder-7B ... (about 30-60 s). Type /exit to quit, /clear to start over.
workspace\llamacpp\llama-cli.exe -m "%MODEL%" -sysf configs\qwen7b-system.txt --temp 0.2 -c 8192
popd
pause
