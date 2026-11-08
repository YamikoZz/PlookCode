@echo off
rem Chat with the 200M SFT model (question -> answer), on CPU so a small GPU won't run out of memory.
pushd "%~dp0"
set RUN=workspace\run-200m-sft
if not exist "%RUN%\best.pt" if not exist "%RUN%\latest.pt" (
  echo.
  echo ยังไม่มีไฟล์โมเดลใน %RUN%
  echo โหลดจาก Google Drive: MyDrive\LocalBrain-200m\run-200m-sft\
  echo   - best.pt  ^(ประมาณ 1.6 GB^)
  echo   - tokenizer.json
  echo วางสองไฟล์ไว้ในโฟลเดอร์ %RUN% แล้วเปิดไฟล์นี้อีกครั้ง
  echo.
  pause
  popd
  exit /b
)
echo กำลังโหลดโมเดล 200M ... ^(ครั้งแรกอาจช้า 1-2 นาที, ตอบแต่ละคำถามใช้เวลาสักครู่^)
start "PlookCode 200M" cmd /k ".venv\Scripts\python.exe -m localbrain chat --run %RUN% --model-only --device cpu"
popd
