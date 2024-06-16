import subprocess

# List of training scripts
scripts = ['unet_v1.0/1st_train.py', 'unet_v1.0/2nd_clean_train.py', 'unet_v1.0/2nd_augmented_train.py']

# Function to run a script
def run_script(script):
    result = subprocess.run(['python', script], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script}: {result.stderr}")
    else:
        print(f"Finished running {script}")

# Loop through the scripts and run them one by one
for script in scripts:
    run_script(script)
