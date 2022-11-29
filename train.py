import os
import torch
import torch.nn as nn
# import matplotlib.pyplot as plt
from numba import cuda
from vgg16 import VGG16_CUSTOM_NET, VGG16_ORI_NET
from module import reset_weights

def train_dataset(train_loader, val_loader, drivePath, type):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    # Check that memory is empty
    lr=1e-4
    num_epochs=50

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = None
    if (type == 'custom'):
        model = VGG16_CUSTOM_NET()
    else:
        model = VGG16_ORI_NET()
    model = model.to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)

    for epoch in range(num_epochs): #I decided to train the model for 50 epochs
        loss_var = 0
        for idx, (images, labels) in enumerate(train_loader):
            print("\n")
            print('[INFO] run batch: {}/{}; epoch: {}/{};'.format(idx, len(train_loader), epoch+1, num_epochs))
            images = images.to(device=device)
            labels = labels.to(device=device)
            ## Forward Pass
            print("[INFO] torch max memory allocated: {} GB".format(round(torch.cuda.max_memory_allocated()/1024**3,1)))
            print("[INFO] torch max memory reserved: {} GB".format(round(torch.cuda.max_memory_reserved()/1024**3,1)))
            print("[INFO] torch memory allocated: {} GB".format(round(torch.cuda.memory_allocated()/1024**3,1)))
            print("[INFO] torch memory reserved: {} GB".format(round(torch.cuda.memory_reserved()/1024**3,1)))        
            print(images.shape)
            optimizer.zero_grad() 
            scores = model(images)      
            loss = criterion(scores,labels)            
            loss.backward()            
            optimizer.step()            
            loss_var += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}] || Step [{idx+1}/{len(train_loader)}] || Loss:{loss_var/len(train_loader)} || Loss Item:{loss.item()}')
            print("\n")
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
        if (type == 'custom'):
            model_path = os.path.join(model_path, "custom")
        else:
            model_path = os.path.join(model_path, "ori")
        cifar_path = os.path.join(model_path, "vgg16_model_{}_epoch_{}.pth".format(type, epoch + 1))
        print("[INFO] save model vgg16_model_epoch_{}.pth".format(epoch + 1))
        torch.save(model.state_dict(), cifar_path) #SAVES THE TRAINED MODEL
        print("\n")

    if (type == 'custom'):
        model = VGG16_CUSTOM_NET()
    else:
        model = VGG16_ORI_NET() 
    model.load_state_dict(torch.load(cifar_path)) #loads the trained model
    model.eval()