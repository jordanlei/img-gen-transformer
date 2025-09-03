# Steerable Image Generation Transformer

A PyTorch-based **Steerable Vision Transformer Variational Autoencoder (VAE)**. In other words, we created a VAE that can take an optional input which specifies *which* class should be generated from a given latent embedding.

![Training Progress](animation_steerable.gif)

## Overview

This project implements a **Steerable TransformerVAE** that combines Vision Transformer architecture with Variational Autoencoder principles. One key component is the ability to **steer** image generation by providing class labels, allowing you to generate images of specific classes (e.g., "generate a 7" or "generate a 3").

The model learns to:
- Encode images into latent representations with class information
- Decode latent vectors back to images conditioned on class labels
- Generate new images of specific classes by providing class cues
- Maintain consistency between intended class and generated image

## Approach

The steerable VAE optimizes a multi-objective loss function that combines four key components:

### Loss Function

The total loss combines four components with a 10× weight on the steering loss:

**L_total = L_recon + L_KL + L_class + 10 × L_steer**

#### 1. Reconstruction Loss
**L_recon = (1/N) × Σ||x_i - x̂_i||²**

Ensures accurate reconstruction of input images.

#### 2. KL Divergence Loss  
**L_KL = -(1/2N) × Σ(1 + log(σ²) - μ² - σ²)**

Regularizes the latent space to follow a standard normal distribution.

#### 3. Classification Loss
**L_class = -(1/N) × Σ y_ic × log(p_ic)**

Trains the encoder to correctly classify input images using cross-entropy.

#### 4. Steering Loss
**L_steer = -(1/N) × Σ ỹ_ic × log(p̃_ic)**

Where ỹ are random class labels and p̃ are the predicted classes of generated images. This enforces consistency between intended and generated classes, making the VAE steerable.

### Teacher Forcing

During training, the model uses ground truth class labels for stable learning. During inference, it uses its own predictions, enabling controlled generation of specific classes.

## Architecture

### Key Components

- **TransformerEncoder**: Converts full images to latent vectors and class predictions using patch embeddings and transformer layers
- **TransformerDecoder**: Converts latent vectors back to full images conditioned on class labels using transformer layers and patch decoding  
- **TransformerVAE**: Main model that combines encoder and decoder with VAE sampling and class conditioning
- **Steering Mechanism**: Novel loss function that enforces class consistency between intended and generated images

### Design Principles

- **Class-Conditioned Generation**: Decoder accepts class labels to steer image generation
- **Teacher Forcing**: Uses ground truth class labels during training for stable learning
- **Consistency Loss**: Steering loss ensures generated images match their intended classes
- **VAE Integration**: Proper variational autoencoder with reconstruction, KL divergence, classification, and steering losses
- **Patch-based Processing**: Efficient image processing using configurable patch sizes
- **Self-attention**: Multi-head attention mechanisms for capturing image dependencies

## Features

- ✅ **Class-Conditioned Generation**: Generate images of specific classes by providing class labels
- ✅ **Steering Loss**: Novel loss function that enforces consistency between intended and generated classes
- ✅ **Teacher Forcing**: Stable training using ground truth class labels
- ✅ **Multi-Objective Training**: Combines reconstruction, KL divergence, classification, and steering losses
- ✅ **Working Training Pipeline**: Complete training loop with MNIST dataset
- ✅ **Progress Visualization**: Automatic generation of training progress plots showing class-specific generation
- ✅ **GIF Animation**: Creates animated GIF demonstrating the model's ability to generate specific digit classes
- ✅ **Model Persistence**: Save/load functionality with metadata preservation
- ✅ **Device Support**: CPU, CUDA, and Apple Silicon (MPS) support
- ✅ **Flexible Dimensions**: Configurable image sizes, patch sizes, and model dimensions

## Quick Start

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd img-gen-transformer

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train the model on MNIST dataset
python train.py
```

This will:
- Download MNIST dataset automatically
- Train the TransformerVAE for 10 epochs
- Generate progress plots every 100 steps
- Create an animated GIF showing training progress
- Save the trained model as `model.pth`

### Model Configuration

The current implementation uses:
- **Dataset**: MNIST (28×28 grayscale images, 10 classes)
- **Model**: 128 embedding dim, 4 attention heads, 4 transformer layers
- **Patches**: 4×4 patch size
- **Training**: Adam optimizer, learning rate 2e-4, batch size 128
- **Loss Weights**: Reconstruction + KL + Classification + 10×Steering loss
- **Teacher Forcing**: Enabled during training for stable class conditioning

## Usage

### Basic Model Creation

```python
from network import TransformerVAE

model = TransformerVAE(
    embed_dim=128,        # Embedding dimension
    num_channels=1,       # Grayscale images
    num_heads=4,          # Number of attention heads
    num_layers=4,         # Number of transformer layers
    patch_size=4,         # Size of image patches
    num_classes=10,       # Number of classes (MNIST digits 0-9)
    image_size=(28, 28),  # Input image dimensions
    teacher_forcing=True  # Enable teacher forcing during training
)
```

### Training

```python
from runner import Runner
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
runner = Runner(model, optimizer, device)

# Train the model
runner.train(train_loader, epochs=10)
```

### Generation

```python
# Generate new images from random latent vectors (all classes)
generated_images = runner.generate(num_samples=4)

# Generate images of specific classes
import torch
import torch.nn.functional as F

# Create one-hot vectors for specific classes
class_0 = torch.eye(10)[0:1]  # Generate digit 0
class_7 = torch.eye(10)[7:8]  # Generate digit 7

# Generate specific digits
z = torch.randn(1, model.embed_dim, device=device)
digit_0 = model.decode(z, class_0)
digit_7 = model.decode(z, class_7)
```

### Model Persistence

```python
# Save model
model.save("model.pth")

# Load model
new_model = TransformerVAE(...)
new_model.load("model.pth")
```

## Project Structure

```
img-gen-transformer/
├── network.py           # Core model architecture
├── runner.py            # Training and generation utilities
├── train.py             # Main training script
├── requirements.txt     # Python dependencies
├── animation.gif        # Training progress animation
└── README.md           # This file
```

## Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **numpy**: Numerical computing
- **tqdm**: Progress bars
- **matplotlib**: Plotting and visualization
- **PIL**: Image processing

## Results

The steerable VAE successfully trains on MNIST digits, learning to generate class-consistent images. The training progress shows:

- **Multi-objective optimization**: Reconstruction, KL divergence, classification, and steering losses all decrease over time
- **Class conditioning**: The model learns to generate images that match their intended classes
- **Steering effectiveness**: The steering loss ensures generated images are correctly classified as their intended class
- **Visual progress**: The animated GIF demonstrates the model's ability to generate specific digit classes (0-9) as training progresses

The key innovation is the **steering loss**, which enforces consistency between the intended class and the generated image, making the VAE truly "steerable" by class labels.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

Copyright (c) 2025 Jordan Lei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
