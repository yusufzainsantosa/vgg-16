import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from train import train_dataset

drivePath = os.getcwd()
print('drivePath: ', drivePath)

BATCH_SIZE = 50

# specify path to the dataset
TRAIN_PATH = os.path.join(drivePath, "dataset", "Train")
VALIDATE_PATH = os.path.join(drivePath, "dataset", "Validate")

# initialize our data augmentation functions
resize = transforms.Resize((224, 224))
hFlip = transforms.RandomHorizontalFlip(p=1)

# initialize our training and validation set data augmentation
# pipeline
datasetTransforms = transforms.Compose([resize, transforms.ToTensor()])

# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
trainDataset = ImageFolder(root=TRAIN_PATH, transform=datasetTransforms)
validateDataset = ImageFolder(root=VALIDATE_PATH, transform=datasetTransforms)
print("[INFO] train dataset contains {} samples...".format(len(trainDataset)))
print("[INFO] validate dataset contains {} samples...".format(len(validateDataset)))

train_loader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(validateDataset, batch_size=BATCH_SIZE)
print("[INFO] train loader length={}".format(len(train_loader)))
print("[INFO] validate loader length={}".format(len(val_loader)))
print("\n")

# display_sample(trainDataset, validateDataset, train_loader, val_loader)
# print('[INFO] start vgg custom training')
# train_dataset(train_loader, val_loader, drivePath, 'custom')
# print("\n")
print('[INFO] start vgg ori training')
train_dataset(train_loader, val_loader, drivePath, 'ori')