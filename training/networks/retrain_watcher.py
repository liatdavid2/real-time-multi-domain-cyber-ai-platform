
import time
import subprocess

while True:
    print("Triggering retraining...")
    subprocess.run(["python", "train.py"])
    time.sleep(60)
