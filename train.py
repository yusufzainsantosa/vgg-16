import os
import torch
import torch.nn as nn
from vgg16 import VGG16_NET

def train_dataset(train_loader, val_loader, drivePath):
  lr=1e-4
  num_epochs=5

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
  model = VGG16_NET() 
  model = model.to(device=device)
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr= lr) 

  for epoch in range(num_epochs): #I decided to train the model for 50 epochs
      loss_var = 0
      
      for idx, (images, labels) in enumerate(train_loader):
          images = images.to(device=device)
          labels = labels.to(device=device)
          ## Forward Pass
          optimizer.zero_grad()
          scores = model(images)
          loss = criterion(scores,labels)
          loss.backward()
          optimizer.step()
          loss_var += loss.item()
          if idx%32==0:
              print(f'Epoch [{epoch+1}/{num_epochs}] || Step [{idx+1}/{len(train_loader)}] || Loss:{loss_var/len(train_loader)}')
      print(f"Loss at epoch {epoch+1} || {loss_var/len(train_loader)}")

      with torch.no_grad():
          correct = 0
          samples = 0
          for idx, (images, labels) in enumerate(val_loader):
              images = images.to(device=device)
              labels = labels.to(device=device)
              outputs = model(images)
              _, preds = outputs.max(1)
              correct += (preds == labels).sum()
              samples += preds.size(0)
          print(f"accuracy {float(correct) / float(samples) * 100:.2f} percentage || Correct {correct} out of {samples} samples")
  
  model_path = os.path.join(drivePath, "model")
  cifar_path = os.path.join(model_path, "cifar10_vgg16_custom_model.pth")
  torch.save(model.state_dict(), cifar_path) #SAVES THE TRAINED MODEL
  model = VGG16_NET()
  model.load_state_dict(torch.load(cifar_path)) #loads the trained model
  model.eval()