"""Helper: set DLL path + HF offline, then run emotyc_predict.py with forwarded args."""
import os, sys

# Fix torch CUDA DLL loading on Windows
_torch_lib = os.path.join(sys.prefix, "Lib", "site-packages", "torch", "lib")
if os.path.isdir(_torch_lib):
    os.add_dll_directory(_torch_lib)
    os.environ["PATH"] = _torch_lib + os.pathsep + os.environ.get("PATH", "")

# Force HuggingFace offline mode (model already cached)
os.environ["HF_HUB_OFFLINE"] = "1"

# Pre-import torch so DLLs are loaded before runpy re-enters the script
import torch

# Forward all CLI args to emotyc_predict.py
import runpy
script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "emotyc_predict.py")
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
