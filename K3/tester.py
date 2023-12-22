import sys
import time
import subprocess
import argparse
import io
from contextlib import redirect_stdout


def _get_index(script_path):
    parts = script_path.split("-")[2:5]
    return "-".join(parts)[:-3]

def _run_execution_command(command):
    original_stdout = sys.stdout
    process = subprocess.Popen(command, shell=True, stdout=sys.stdout, stderr=subprocess.PIPE, text=True)
    process.communicate()
    if process.returncode != 0:
        print(f"No output!")
    
    sys.stdout = original_stdout
    sys.stdout = sys.__stdout__

def execute_script(script_path, data_path):
    print(f"*** Executing script {script_path}")

    index = _get_index(script_path)    
    start_time = time.time()
    command = f"python {script_path} {data_path}"
    result = _run_execution_command(command)

    duration = time.time() - start_time
    print(f"*** Index: {index} - Result: OK - Duration: {duration} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", help="Path to the script")
    parser.add_argument("data_path", help="Path to the dataset")

    args = parser.parse_args()

    execute_script(args.script_path, args.data_path)
