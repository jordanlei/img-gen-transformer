import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class Runner:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.random_latents = torch.randn(12, self.model.embed_dim, device=self.device)
    
    def train(self, train_loader, epochs):
        self.step = 0
        progress_bar = tqdm(range(epochs), desc="Training")
        self.model.train()
        for epoch in progress_bar:
            for batch in train_loader:
                x, _ = batch

                if x.dim() == 3:  # If image is [B, H, W]
                    x = x.unsqueeze(1)  # Add channel dimension [B, C, H, W]

                x = x.to(self.device)
                self.optimizer.zero_grad()
                
                # Now the model returns (recon_x, mu, logvar) as intended
                recon_x, mu, logvar = self.model(x)

                recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
                total_loss = recon_loss + kl_loss
                total_loss.backward()
                self.optimizer.step()
                
                if self.step % 100 == 0:
                    progress_bar.set_description(f"Epoch {epoch} - Loss: {total_loss.item():.4f}")
                    self.plot(f"temp/step_{self.step:08d}.png")
                self.step += 1

    def generate(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            # Sample from the learned latent distribution
            z = torch.randn(num_samples, self.model.embed_dim, device=self.device)
            recon_x = self.model.decode(z)
            return recon_x

    def plot(self, save_path):
        # Generate samples
        samples = self.model.decode(self.random_latents)
        
        # Untransform the normalized values back to [0,1] range
        # Assuming normalization was done with mean=0.5, std=0.5
        samples = samples * 0.5 + 0.5
        
        # Create a figure with 3x4 subplot grid
        fig = plt.figure(figsize=(12, 9))
        for i in range(12):
            ax = fig.add_subplot(3, 4, i+1)
            
            # Get sample and move to CPU if needed
            img = samples[i].cpu().detach().numpy()
            
            # Preserve channel info by checking number of channels
            if img.shape[0] == 1:  # Single channel
                ax.imshow(img.squeeze(), cmap='gray')
            else:  # RGB/multiple channels
                # Rearrange from CxHxW to HxWxC
                img = img.permute(1, 2, 0)
                ax.imshow(img)
                
            ax.axis('off')
        plt.suptitle(f"Step {self.step}")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
