import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

phf_kernel =  np.array([[[0., 0., 0., 0., 0.],
                        [0., 0., -1., 0., 1.],
                        [0., -1., 0., 1., 0.],
                        [-1., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0.]],

                        [[0., 0., 0., 0., 0.],
                        [0., 0., -1., 1., 0.],
                        [0., -1., 1., 0., 0.],
                        [-1., 1., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]],
                        
                        [[0., 0., 0., 0., 0.],
                        [0., 0., -1., -1., -1.],
                        [0., 0., 0., 0., 0.],
                        [1., 1., 1., 0., 0.],
                        [0., 0., 0., 0., 0.]],
                        
                        [[0., 0., 0., 0., 0.],
                        [0., 0., -1., -1., -1.],
                        [0., 1., 1., 1., 0.],
                        [0., 0., 0., 0., 0.],
                        [0., 0., 0., 0., 0.]],
                        
                        [[0., 0., 0., 0., 0.],
                        [0., 0., 1., 0., -1.],
                        [0., -1., 0., -1., 0.],
                        [-1., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0.]]])
                                       
custom_kernel = np.array([[0., 0., 0., 0., 0.],
                        [0., 0., -1., 0., 1.],
                        [0., -1., 0., 1., 0.],
                        [-1., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 0.]])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

class VGG16_NET(nn.Module):
    def __init__(self):
        super(VGG16_NET, self).__init__()

        custom_kernel = None
        for phf in phf_kernel:
          phf_flip = np.fliplr(phf)
          rot = np.rot90(phf, 3)
          rot_flip = np.fliplr(rot)
          
          if (custom_kernel is None):
            custom_kernel = np.array([phf])
          else:
            custom_kernel = np.concatenate([custom_kernel, [phf]], axis=0)
          custom_kernel = np.concatenate([custom_kernel, [phf_flip]], axis=0)
          custom_kernel = np.concatenate([custom_kernel, [rot]], axis=0)
          custom_kernel = np.concatenate([custom_kernel, [rot_flip]], axis=0)

        weights = torch.from_numpy(custom_kernel)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            weights_conv1 = weights.view(20, 1, 5, 5).repeat(1, 3, 1, 1)
            weights_conv1 = weights_conv1.to(torch.float32)
            self.conv1_5.weight = nn.Parameter(weights_conv1)

        self.conv2 = nn.Conv2d(in_channels=84, out_channels=64, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv2d(in_channels=84, out_channels=20, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            weights_conv2 = weights.view(20, 1, 5, 5).repeat(1, 84, 1, 1)
            weights_conv2 = weights_conv2.to(torch.float32)
            self.conv2_5.weight = nn.Parameter(weights_conv2)

        self.conv3 = nn.Conv2d(in_channels=84, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc14 = nn.Linear(25088, 4096)
        self.fc15 = nn.Linear(4096, 4096)
        self.fc16 = nn.Linear(4096, 10)

    def forward(self, x):
        x_ori_conv1 = F.relu(self.conv1(x))
        x_custom_conv1 = F.relu(self.conv1_5(x))
        x = torch.cat((x_ori_conv1, x_custom_conv1), dim=1)

        x_ori_conv2 = F.relu(self.conv2(x))
        x_custom_conv2 = F.relu(self.conv2_5(x))
        x = torch.cat((x_ori_conv2, x_custom_conv2), dim=1)

        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = self.maxpool(x)
        x = F.relu(self.conv8(x))
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = self.maxpool(x)
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.relu(self.conv13(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5) #dropout was included to combat overfitting
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = self.fc16(x)
        return x