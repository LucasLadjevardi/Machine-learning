import os
from PIL import Image
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.optim import Adam

# Define a simple CNN model :: Change accordingly
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
class BlurredSharpDatasetNew(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.blurred_images = os.listdir(os.path.join(root_dir, 'blur'))
        self.sharp_images = os.listdir(os.path.join(root_dir, 'sharp'))

    def __len__(self):
        return len(self.blurred_images)

    def __getitem__(self, idx):
        blurred_img_name = os.path.join(self.root_dir, 'blur', self.blurred_images[idx])
        sharp_img_name = os.path.join(self.root_dir, 'sharp', self.sharp_images[idx])

        blurred_img = Image.open(blurred_img_name).convert('RGB')
        sharp_img = Image.open(sharp_img_name).convert('RGB')

        if self.transform:
            blurred_img = self.transform(blurred_img)
            sharp_img = self.transform(sharp_img)

        return blurred_img, sharp_img

# Define transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def main():
    
    # Define dataset
    dataset = BlurredSharpDatasetNew(root_dir="data", transform=transform)
    datasetvaild = BlurredSharpDatasetNew(root_dir="Validation", transform=transform)

    # Define data loader
    batch_size = 36
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=7)
    dataloaderValid = DataLoader(datasetvaild, batch_size=batch_size, shuffle=False, num_workers=7)

    # Define model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SharpeningModelAdvanced().to(device)
    if os.path.exists("sharpening_model.pth"):
        model.load_state_dict(torch.load("sharpening_model.pth"))
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 1
    minimalloss = numpy.inf
    for epoch in range(num_epochs):
        model.train()
        for i, (blurred_img, sharp_img) in enumerate(dataloader):
            blurred_img, sharp_img = blurred_img.to(device), sharp_img.to(device)
            optimizer.zero_grad()
            outputs = model(blurred_img)
            loss = criterion(outputs, sharp_img)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

        model.eval()
        validationloss = 0

        with torch.no_grad():
            for inputs, target in dataloaderValid:
                inputs, target = inputs.to(device), target.to(device)
                deblurred_img=model(inputs)
                temploss=criterion(deblurred_img, target)
                validationloss+=temploss.item()

        print(f"Validation loss [{validationloss/len(dataloaderValid):.4f}]")
        if validationloss < minimalloss:
            minimalloss=validationloss
            torch.save(model.state_dict(), "sharpening_model.pth")

if __name__ == '__main__':
    main()