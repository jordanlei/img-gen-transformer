import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(in_features=64*8*8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*8*8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiHeadAttention(nn.Module):
    # compute multi-head attention in a vectorized fashion
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.W_qkv = nn.Linear(embed_dim, 3*embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
    
    def forward(self, x):
        # compute q, k, v in a vectorized fashion
        qkv = self.W_qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim = -1) # shape: (batch_size, seq_len, embed_dim) each

        batch_size, seq_len, embed_dim = q.shape
        head_dim = embed_dim // self.num_heads

        # reshape q, k, v to (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_dim).transpose(1, 2)
        
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        self.attn_weights = F.softmax(attn_weights, dim = -1)
        attn_output = torch.matmul(self.attn_weights, v) #shape: (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.W_o(attn_output)
    
    def save(self, path):
        metadata = {
            "num_heads": self.num_heads,
            "embed_dim": self.embed_dim
        }
        torch.save({
            "model_state_dict": self.state_dict(),
            "metadata": metadata
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.num_heads = checkpoint["metadata"]["num_heads"]
        self.embed_dim = checkpoint["metadata"]["embed_dim"]
        return self

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
    
    def forward(self, x):
        # pre-norm is a design choice
        x = x + self.attn(self.norm(x))
        x = x + self.mlp(self.norm(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, patch_size, num_channels, image_size = (28, 28)):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_h = image_size[0] // patch_size
        self.patch_w = image_size[1] // patch_size
        
        # Calculate number of patches
        self.num_patches = self.patch_h * self.patch_w
        
        # Patch embedding - converts image to patch embeddings
        self.patch_embed = nn.Conv2d(num_channels, embed_dim, patch_size, patch_size)
        
        # Learnable position embedding for patches + CLS token
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.num_patches, embed_dim))
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output heads for VAE
        self.fc_mu = nn.Linear(embed_dim, embed_dim)
        self.fc_logvar = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, x):
        # Input: (batch_size, num_channels, image_height, image_width) - full image
        # Output: (batch_size, embed_dim), (batch_size, embed_dim) - mu and logvar
        
        batch_size = x.shape[0]
        
        # Validate input dimensions
        assert x.shape[2] == self.image_size[0] and x.shape[3] == self.image_size[1], \
            f"Expected image size {self.image_size}, got {x.shape[2:]}"
        
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.norm(x)
        
        # Take CLS token for VAE encoding
        cls_output = x[:, 0, :]  # (batch_size, embed_dim)
        
        # Generate VAE parameters
        mu = self.fc_mu(cls_output)
        logvar = self.fc_logvar(cls_output)
        
        return mu, logvar

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, patch_size, num_channels, image_size = (28, 28)):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_h = image_size[0] // patch_size
        self.patch_w = image_size[1] // patch_size
    
        # Calculate number of patches
        self.num_patches = self.patch_h * self.patch_w
        
        # Learnable position embedding for patches (no CLS token needed for generation)
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Patch reverse - converts patch embeddings back to image
        self.patch_reverse = nn.ConvTranspose2d(embed_dim, num_channels, patch_size, patch_size)
    
    def forward(self, z):
        # Input: (batch_size, embed_dim) - latent vector
        # Output: (batch_size, num_channels, image_height, image_width) - reconstructed image
        batch_size = z.shape[0]
        z = z.unsqueeze(1).repeat(1, self.num_patches, 1)
        z = z + self.pos_embed
        
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)
        
        # Reshape for patch reverse: (batch_size, num_patches, embed_dim) -> (batch_size, embed_dim, H//patch_size, W//patch_size)
        z = z.transpose(1, 2).view(batch_size, self.embed_dim, self.patch_h, self.patch_w)
        
        # Patch reverse: (batch_size, embed_dim, H//patch_size, W//patch_size) -> (batch_size, num_channels, H, W)
        x = self.patch_reverse(z)
        
        return x

class TransformerVAE(nn.Module):
    def __init__(self, embed_dim, num_channels, num_heads, num_layers, patch_size, image_size=(28, 28)):
        super(TransformerVAE, self).__init__()
        self.embed_dim = embed_dim
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.image_size = image_size
        
        # Encoder and decoder now handle full images and patch operations internally
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers, patch_size, num_channels, image_size)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers, patch_size, num_channels, image_size)

    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar
    
    def save(self, path):
        metadata = {
            "patch_size": self.patch_size,
            "image_size": self.image_size,
            "embed_dim": self.embed_dim,
            "num_channels": self.num_channels,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
        }
        torch.save({
            "model_state_dict": self.state_dict(),
            "metadata": metadata
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        
        # Restore metadata
        metadata = checkpoint["metadata"]
        self.patch_size = metadata["patch_size"]
        self.image_size = metadata["image_size"]
        self.embed_dim = metadata["embed_dim"]
        self.num_channels = metadata["num_channels"]
        self.num_heads = metadata["num_heads"]
        self.num_layers = metadata["num_layers"]
        
        return self