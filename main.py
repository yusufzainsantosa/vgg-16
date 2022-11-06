import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from module import display_sample
from train import train_dataset

drivePath = os.getcwd()

BATCH_SIZE = 32

# specify path to the dataset
DATASET_PATH = os.path.join(drivePath, "dataset")

# initialize our data augmentation functions
resize = transforms.Resize((224, 224))
hFlip = transforms.RandomHorizontalFlip(p=1)
# vFlip = transforms.RandomVerticalFlip(p=0.25)
# rotate = transforms.RandomRotation(degrees=15)

# initialize our training and validation set data augmentation
# pipeline
trainTransforms = transforms.Compose([resize, transforms.ToTensor()])
valTransforms = transforms.Compose([resize, hFlip, transforms.ToTensor()])

# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = ImageFolder(root=DATASET_PATH, transform=trainTransforms)
valDataset = ImageFolder(root=DATASET_PATH, transform=valTransforms)
print("[INFO] training dataset contains {} samples...".format(len(trainDataset)))
print("[INFO] validation dataset contains {} samples...".format(len(valDataset)))

# create training and validation set dataloaders
print("[INFO] creating training and validation set dataloaders...")
train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(valDataset, batch_size=BATCH_SIZE)

display_sample(trainDataset, valDataset, train_loader, val_loader)

train_dataset(train_loader, val_loader, drivePath)