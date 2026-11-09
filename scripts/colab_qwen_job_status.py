import os
import subprocess

commands = """
echo PROCESS
pgrep -af '[c]olab_qwen_refusal_lora' || true
echo GPU
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo LOG
tail -n 30 /content/qwen-lora.log 2>/dev/null || tail -n 30 /content/qwen-lora-launch.log 2>/dev/null || true
echo OUTPUTS
find /content/qwen-authorized-lora -maxdepth 2 -type f -printf '%p %s bytes\n' 2>/dev/null | tail -n 20 || true
echo CACHE
du -sh /content/hf-cache 2>/dev/null || true
ps -o pid,etime,%cpu,%mem,cmd -p $(pgrep -f '[c]olab_qwen_refusal_lora.py' | head -n1) 2>/dev/null || true
"""
result = subprocess.run(commands, shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
