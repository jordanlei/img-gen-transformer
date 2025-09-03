import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

class Runner:
    def __init__(self, model, optimizer, device, use_class_loss = True, use_steer_loss = True, plot_dir = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.use_class_loss = use_class_loss
        self.use_steer_loss = use_steer_loss

        if plot_dir: os.makedirs(plot_dir, exist_ok=True)
        self.plot_dir = plot_dir

        # for plotting purposes
        self.random_latents = torch.randn(10, self.model.embed_dim, device=self.device)

    def one_hot(self, y): 
        return torch.eye(self.model.num_classes, device=self.device)[y]
    
    def train(self, train_loader, epochs):
        self.step = 0
        progress_bar = tqdm(range(epochs), desc="Training")
        self.model.train()
        for epoch in progress_bar:
            for batch in train_loader:
                x, y = batch

                if x.dim() == 3:  # If image is [B, H, W]
                    x = x.unsqueeze(1)  # Add channel dimension [B, C, H, W]

                x = x.to(self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()

                # if we are teacher forcing, we input the one-hot encoded class y
                # otherwise, we let the network predict the class itself
                # cls_input = self.one_hot(y) if self.teacher_forcing else None
                recon_x, mu, logvar, cls_output = self.model(x, None)

                # VAE losses
                recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
                class_loss = F.cross_entropy(cls_output, y) if self.use_class_loss else torch.tensor(0.0)

                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std

                random_classes = torch.randint(0, self.model.num_classes, (x.shape[0],), device=self.device)
                random_cls_input = self.one_hot(random_classes)
                steer_recon = self.model.decode(z.detach(), random_cls_input)
                _, _, steer_cls_output = self.model.encode(steer_recon)
                steer_loss = F.cross_entropy(steer_cls_output, random_classes) if self.use_steer_loss else torch.tensor(0.0)

                # Total loss
                total_loss = recon_loss + kl_loss + 500 * class_loss + steer_loss
                total_loss.backward()
                self.optimizer.step()
                
                if self.step % 100 == 0:
                    progress_bar.set_description(
                        f"Epoch {epoch} - total: {total_loss.item():.4f} | recon: {recon_loss.item():.2f} | "
                            f"KL: {kl_loss.item():.2f} | class: {class_loss.item():.2f} | steer: {steer_loss.item():.2f}"
                    )
                    if self.plot_dir: 
                        self.plot(f"{self.plot_dir}/step_{self.step:08d}.png")
                self.step += 1

    def generate(self, num_samples):
        self.model.eval()
        with torch.no_grad():
            # Sample from the learned latent distribution
            z = torch.randn(num_samples, self.model.embed_dim, device=self.device)
            recon_x = self.model.decode(z, self.cls_input)
            return recon_x

    def plot(self, save_path):
        # Generate samples
        samples = self.model.decode(self.random_latents, self.one_hot(torch.arange(10, device=self.device)))
        
        # Untransform the normalized values back to [0,1] range
        # Assuming normalization was done with mean=0.5, std=0.5
        samples = samples * 0.5 + 0.5
        
        # Create a figure with 3x4 subplot grid
        fig = plt.figure(figsize=(12, 9))
        for i in range(10):
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
            ax.set_title(f"Class {i}")
        plt.suptitle(f"Step {self.step}")
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, load_path):
        with open(load_path, 'rb') as f:
            self = pickle.load(f)
        return self
