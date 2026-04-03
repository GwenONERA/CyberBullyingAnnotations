"""Wrapper to fix torch DLL loading on Windows before running emotyc_predict."""
import os
import sys

# Fix DLL loading for torch CUDA on Windows
torch_lib = os.path.join(
    os.path.dirname(sys.executable), '..', 'Lib', 'site-packages', 'torch', 'lib'
)
torch_lib = os.path.abspath(torch_lib)
if os.path.isdir(torch_lib):
    os.add_dll_directory(torch_lib)

# Now run the actual script
sys.argv[0] = os.path.join(os.path.dirname(__file__), 'emotyc_predict.py')
exec(open(sys.argv[0], encoding='utf-8').read())
