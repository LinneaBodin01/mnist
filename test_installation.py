import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def test_pytorch():
    try:
        # Check if PyTorch is installed and print the version
        print("PyTorch Version:", torch.__version__)

        # Test a basic PyTorch operation
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        z = x + y
        print("PyTorch Addition Result:", z)

        # You can add more specific PyTorch tests here
    except Exception as e:
        print("PyTorch test failed:", str(e))


def show_random_img(path, grid_size):
    transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = datasets.MNIST(
        root=path, train=True, transform=transform, download=True
    )
    dataloader = DataLoader(mnist_dataset, batch_size=1, shuffle=True)

    sns.set_style("darkgrid")
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))

    # Randomly select and plot X*X images
    for i in range(grid_size):
        for j in range(grid_size):
            random_idx = np.random.randint(len(mnist_dataset))
            image, label = mnist_dataset[random_idx]

            # Apply Seaborn style to Matplotlib plot
            sns.heatmap(image[0], cmap="gray", ax=axes[i, j], cbar=False)
            axes[i, j].set_title(f"Label: {label}")
            axes[i, j].axis("off")
            
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    print(len(mnist_dataset), "images loaded")
    plt.show()
