import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import os
import torch
import torchvision.transforms as transforms
from PIL import Image


class SharpeningModel(nn.Module):
    def __init__(self):
        super(SharpeningModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class SharpeningModelAdvanced(nn.Module):
    def __init__(self):
        super(SharpeningModelAdvanced, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Define custom dataset for loading blurred and sharp image pairs
class BlurredSharpDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.blurred_dataset = ImageFolder(root=root_dir, transform=self.transform)
        self.sharp_dataset = ImageFolder(root=root_dir, transform=self.transform)

    def __len__(self):
        return len(self.blurred_dataset)

    def __getitem__(self, idx):
        blurred_img, _ = self.blurred_dataset[idx]
        sharp_img, _ = self.sharp_dataset[idx]
        return blurred_img, sharp_img

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SharpeningModelAdvanced().to(device)
    if os.path.exists("sharpening_model.pth"):
        print("hello")
        model.load_state_dict(torch.load("sharpening_model.pth"))
    model.eval()

    def process_and_save_image(image_path, output_path):
        image = Image.open(image_path)
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            deblurred_img = model(image_tensor)

        deblurred_img = transforms.ToPILImage()(deblurred_img.squeeze(0).cpu())
        deblurred_img.save(output_path)

    input_image_path = "0_blur.jpg"
    output_image_path = "output_image.jpg"
    process_and_save_image(input_image_path, output_image_path)

if __name__ == '__main__':
    main()