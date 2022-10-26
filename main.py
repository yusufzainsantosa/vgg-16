import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from module import visualize_batch
from train import train_dataset

drivePath = os.getcwd()

BATCH_SIZE = 64

workers = 4
class_size = 10

# specify path to the dataset
DATASET_PATH = os.path.join(drivePath, "dataset")

# initialize our data augmentation functions
resize = transforms.Resize((422, 422))
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

# grab a batch from both training and validation dataloader
trainIter = iter(train_loader)
valIter = iter(val_loader)
trainBatch = next(trainIter)
valBatch = next(valIter)
# visualize the training and validation set batches
print("[INFO] visualizing training and validation batch...")
visualize_batch(trainBatch, trainDataset.classes, "train")
visualize_batch(valBatch, valDataset.classes, "val")

for images, _ in train_loader:
    print('images.shape:', images.shape)
    plt.figure(figsize=(16,8))
    plt.axis('off')
    plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    break

sample_image=iter(train_loader)
samples,labels=sample_image.next()
print(samples.shape) #64 batch size, 1 channel, width 224 , height 224
print(labels)

print(len(train_loader))

train_dataset(train_loader, val_loader, drivePath)