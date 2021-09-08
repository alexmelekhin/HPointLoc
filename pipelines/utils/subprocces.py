# Copyright 2020-present NAVER Corp. Under BSD 3-clause license
import subprocess
import sys
import os.path as path
from typing import List, Optional

def run_python_command(local_path: str, args: List[str], python_binary: Optional[str] = None, notbash = True):
    """
    run a python subprocess
    :param local_path: path where you expect the file to be
    :type local_path: str
    :param args: the arguments of the python process
    :type args: List[str]
    :param python_binary: path to the python binary, optional, when None, the .py file is called directly
    :type python_binary: Optional[str]
    :raises ValueError: subprocess crashed
    """
    if python_binary is None and notbash:
        if path.isfile(local_path):
            compute_image_pairs_bin = path.normpath(local_path)
        else:
            # maybe the script was installed through pip
            compute_image_pairs_bin = path.basename(local_path)
        args.insert(0, compute_image_pairs_bin)        

    use_shell = sys.platform.startswith("win")
    sub_process = subprocess.Popen(args, shell=use_shell)
    sub_process.wait()
    if sub_process.returncode != 0:
        raise ValueError('\nSubprocess Error (Return code:' f' {sub_process.returncode} )')