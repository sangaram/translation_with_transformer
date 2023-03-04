import os
import sys

"""
Setup file to create usefull environment variables
"""
config = "\n\n# Project environment variables\n"
if len(sys.argv) == 1:
    raise Exception("Error, you must specify in arguments the environment name. Usage: ./setup.py env_name")

env_name = sys.argv[1]

ROOT = os.getcwd() # Project root direcoty environment variable
config = f"{config}\nexport ROOT={ROOT}"

# Saving all the environment variables
with open(os.path.join(ROOT, f"{env_name}/bin/activate"), "a") as f:
    f.write(config)
