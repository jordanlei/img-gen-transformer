# Image Generation Transformer

A PyTorch-based project for vision transformers and image generation.

## Setup

### Virtual Environment

This project uses a Python virtual environment. To activate it:

```bash
source .venv/bin/activate
```

To deactivate:
```bash
deactivate
```

### Installation

All required packages are already installed in the virtual environment. If you need to reinstall them:

```bash
pip install -r requirements.txt
```

## Installed Packages

### Core PyTorch
- **torch**: PyTorch deep learning framework
- **torchvision**: Computer vision utilities for PyTorch
- **torchaudio**: Audio processing utilities for PyTorch

### Data Science & Visualization
- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization

### Development Tools
- **jupyter**: Jupyter notebook environment
- **ipykernel**: IPython kernel for Jupyter
- **tqdm**: Progress bars for loops

### Additional Dependencies
- **pillow**: Image processing (PIL)
- **sympy**: Symbolic mathematics
- **networkx**: Graph theory and network analysis

## Project Structure

```
img-gen-transformer/
├── .venv/                 # Virtual environment
├── .gitignore            # Git ignore file
├── requirements.txt       # Python dependencies
└── README.md            # This file
```

## Usage

1. Activate the virtual environment: `source .venv/bin/activate`
2. Start Jupyter: `jupyter lab` or `jupyter notebook`
3. Or run Python scripts directly: `python your_script.py`

## Notes

- The virtual environment is set up with Python 3.13
- PyTorch is installed with CPU support (optimized for macOS ARM64)
- All packages are compatible with the latest stable versions
- The `.gitignore` file is configured to exclude common PyTorch project files like model checkpoints, datasets, and logs
