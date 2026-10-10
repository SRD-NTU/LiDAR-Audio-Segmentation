# Reproducible environment

`requirements.txt` installs the core, portable environment used by all three
methods, including the CLAP audio dependency. `requirements-foundation.txt`
adds optional raw LiDAR foundation geometry support, including a CUDA-specific
sparse-convolution wheel.

The environment files do not bundle a Python virtual environment or downloaded
model weights. Create the environment on the target operating system and select
a PyTorch build compatible with its CPU/CUDA driver. The original development
environment used Python 3.10, PyTorch 2.8, NumPy 1.26, SciPy 1.15, PyYAML 6.0,
scikit-learn 1.7, LightGBM 4.7, and Transformers 4.47.

For a CUDA system, install the appropriate PyTorch wheel first if needed, then
run `python -m pip install -r environment/requirements.txt`. The core LiDAR-WM
and Point-Bind models work without the optional foundation stack.
