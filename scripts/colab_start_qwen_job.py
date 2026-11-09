import subprocess

commands = """
chmod +x /content/colab_qwen_refusal_lora.sh
nohup bash /content/colab_qwen_refusal_lora.sh >/content/qwen-lora-launch.log 2>&1 &
echo $!
"""
result = subprocess.run(commands, shell=True, capture_output=True, text=True)
print(result.stdout)
print(result.stderr)
