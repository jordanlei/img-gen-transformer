from network import TransformerVAE
from runner import Runner
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
import os
import glob
from PIL import Image
import shutil


def write_to_gif(dir = "temp"):
    """Create GIF from saved figures with extended first and last frame durations."""
    print("Creating GIF from training progress figures...")
    try:
        # Get all PNG files in temp_figures directory
        image_files = sorted(glob.glob(f"{dir}/*.png"))
        
        if image_files:
            # Open all images
            images = []
            for filename in image_files:
                img = Image.open(filename)
                images.append(img)
            
            # Create duration list: first and last frames stay longer
            durations = []
            for i in range(len(images)):
                if i == 0:  # First frame
                    durations.append(2000)  # 2 seconds
                elif i == len(images) - 1:  # Last frame
                    durations.append(3000)  # 3 seconds
                else:  # Middle frames
                    durations.append(200)   # 0.5 seconds
            
            # Save as GIF in main directory
            gif_filename = "animation.gif"
            images[0].save(
                gif_filename,
                save_all=True,
                append_images=images[1:],
                duration=durations,
                loop=0
            )
            print(f"GIF created successfully: {gif_filename}")
            print(f"Total frames: {len(images)}")
            print(f"First frame duration: {durations[0]}ms, Last frame duration: {durations[-1]}ms")
            
        else:
            print("No PNG files found to create GIF")
            
    except Exception as e:
        print(f"Error creating GIF: {e}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")


    if os.path.exists("temp"):
        shutil.rmtree("temp")
        print("Removing existing temporary directory")

    # load data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # load model with correct parameters
    model = TransformerVAE(
        embed_dim=16, 
        num_channels=1, 
        num_heads=4, 
        num_layers=5, 
        patch_size=4,
        num_classes=10,
        image_size=(28, 28),
    )
    
    # Move model to device
    model = model.to(device)
    
    # load optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    
    os.makedirs("temp", exist_ok=True)
    runner = Runner(model, optimizer, device)
    runner.train(train_loader, epochs=30)
    write_to_gif()
    model.save("model.pth")
    # Clean up temporary files
    if os.path.exists("temp"):
        shutil.rmtree("temp")
        print("Cleaned up temporary directory")



if __name__ == "__main__":
    main()

    




