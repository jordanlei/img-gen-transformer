from network import TransformerVAE
from runner import Runner
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import os
import glob
from PIL import Image
import shutil

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # load model with correct parameters
    model = TransformerVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    runner = Runner(model, optimizer, device, use_class_loss=False, use_steer_loss=False)
    runner.train(train_loader, epochs=20)
    runner.save("runner_no_class_no_steer.pkl")

    model1 = TransformerVAE().to(device)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=5e-3)
    runner = Runner(model1, optimizer1, device, use_class_loss=True, use_steer_loss=False)
    runner.train(train_loader, epochs=20)
    runner.save("runner_class_no_steer.pkl")

    model2 = TransformerVAE().to(device)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=5e-3)
    runner = Runner(model2, optimizer2, device, use_class_loss=True, use_steer_loss=True)
    runner.train(train_loader, epochs=20)
    runner.save("runner_class_steer.pkl")



if __name__ == "__main__":
    main()

    




