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
    def __init__(self, embed_dim, num_heads, num_layers, num_classes):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.norm = nn.LayerNorm(embed_dim)
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def save(self, path):
        metadata = {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes
        }
        torch.save({
            "model_state_dict": self.state_dict(),
            "metadata": metadata
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.embed_dim = checkpoint["metadata"]["embed_dim"]
        self.num_heads = checkpoint["metadata"]["num_heads"]
        self.num_layers = checkpoint["metadata"]["num_layers"]