import subprocess
import sys
import os

APP_PATH = r"C:\Users\SHRIVAS CHIKRAM\Desktop\snapdragon\app.py"

print("Starting Snapdragon AI Dashboard...")

subprocess.Popen([
    sys.executable,
    "-m",
    "streamlit",
    "run",
    APP_PATH
])

print("Streamlit started.")
print("Open: http://localhost:8501")
