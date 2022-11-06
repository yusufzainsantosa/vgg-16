import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from module import display_sample
from train import train_dataset

drivePath = os.getcwd()

BATCH_SIZE = 25
K_FOLD = 5

# specify path to the dataset
DATASET_PATH = os.path.join(drivePath, "dataset")

# initialize our data augmentation functions
resize = transforms.Resize((224, 224))
hFlip = transforms.RandomHorizontalFlip(p=1)

# initialize our training and validation set data augmentation
# pipeline
datasetTransforms = transforms.Compose([resize, transforms.ToTensor()])

# initialize the training and validation dataset
print("[INFO] loading the training and validation dataset...")
allDataset = ImageFolder(root=DATASET_PATH, transform=datasetTransforms)
print("[INFO] dataset contains {} samples...".format(len(allDataset)))

# split data
step = len(allDataset)//K_FOLD
for k_index in range(K_FOLD):
  left_num = k_index * step
  right_num = left_num + step

  train_list = list(range(0, len(allDataset)))
  valid_list = train_list[left_num:right_num]
  del train_list[left_num:right_num]

  trainset_k5 = Subset(allDataset, train_list)
  validate_k5 = Subset(allDataset, valid_list)

  # create training and validation set dataloaders
  print("[INFO] creating training and validation set dataloaders...")
  print("[INFO] trainset with k=5, delete data from {} to {}".format(left_num, right_num))
  print("[INFO] trainset with k=5, has a length {}".format(len(train_list)))
  print("[INFO] validate with k=5, get data from {} to {}".format(left_num, right_num))
  print("[INFO] validate with k=5, has a length {}".format(len(valid_list)))
  print("\n")

  train_loader = DataLoader(trainset_k5, batch_size=BATCH_SIZE, shuffle=True)
  val_loader = DataLoader(validate_k5, batch_size=BATCH_SIZE)
  print("[INFO] train loader length={}".format(len(train_loader)))
  print("[INFO] validate loader length={}".format(len(val_loader)))
  print("\n")

  # display_sample(allDataset, train_loader)
  train_dataset(train_loader, val_loader, drivePath, k_index)