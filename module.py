#import libraries
import os
import torch
import torch.nn as nn
import shutil
import matplotlib.pyplot as plt

from torchvision.utils import make_grid

BATCH_SIZE_DATASET = 5

def copy_images(imagePaths, folder):
	# check if the destination folder exists and if not create it
	if not os.path.exists(folder):
		os.makedirs(folder)
	# loop over the image paths
	for path in imagePaths:
		# grab image name and its label from the path and create
		# a placeholder corresponding to the separate label folder
		imageName = path.split(os.path.sep)[-1]
		label = path.split(os.path.sep)[-2]
		labelFolder = os.path.join(folder, label)
		# check to see if the label folder exists and if not create it
		if not os.path.exists(labelFolder):
			os.makedirs(labelFolder)
		# construct the destination image path and copy the current
		# image to it
		destination = os.path.join(labelFolder, imageName)
		shutil.copy(path, destination)

def reset_weights(m):
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		m.reset_parameters()

def visualize_batch(batch, classes, dataset_type):
	# initialize a figure
	fig = plt.figure("{} batch".format(dataset_type),
		figsize=(BATCH_SIZE_DATASET, BATCH_SIZE_DATASET))
	# loop over the batch size
	for i in range(0, BATCH_SIZE_DATASET):
		# create a subplot
		ax = plt.subplot(2, 4, i + 1)
		# grab the image, convert it from channels first ordering to
		# channels last ordering, and scale the raw pixel intensities
		# to the range [0, 255]
		image = batch[0][i].cpu().numpy()
		image = image.transpose((1, 2, 0))
		image = (image * 255.0).astype("uint8")
		# grab the label id and get the label from the classes list
		idx = batch[1][i]
		label = classes[idx]
		# show the image along with the label
		plt.imshow(image)
		plt.title(label)
		plt.axis("off")
	# show the plot
	plt.tight_layout()
	plt.show()

def display_sample(allDataset, train_loader):
	# grab a batch from both training and validation dataloader
	trainIter = iter(train_loader)
	trainBatch = next(trainIter)
	print("[INFO] visualizing dataset batch...")
	visualize_batch(trainBatch, allDataset.classes, "train")

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
