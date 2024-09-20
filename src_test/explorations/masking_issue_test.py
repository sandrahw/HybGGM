'''masking issue'''
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import random
random.seed(10)
print(random.random())

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x_conv)
        return x_conv, x_pool

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Center crop the skip connection tensor to match the size of the upsampled tensor
        if x.size() != skip.size():
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)  # Concatenate along channel axis
        x = self.conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.enc1 = EncoderBlock(1, 64)   # Input channels = 21, Output channels = 64
        self.enc2 = EncoderBlock(64, 128)
        # self.enc3 = EncoderBlock(128, 256)
        # self.enc4 = EncoderBlock(256, 512)
        
        # self.bottleneck = ConvBlock(512, 1024)  # Bottleneck layer
        self.bottleneck = ConvBlock(128, 256) 

        # self.dec4 = DecoderBlock(1024, 512)
        # self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)

        self.output_conv = nn.Conv2d(64, 1, kernel_size=1)  # Output layer with 1 channel

    def forward(self, x):
        # Encoder path
        x1, p1 = self.enc1(x)
        x2, p2 = self.enc2(p1)
        # x3, p3 = self.enc3(p2)
        # x4, p4 = self.enc4(p3)

        # Bottleneck
        # b = self.bottleneck(p4)
        b = self.bottleneck(p2)

        # Decoder path
        # d4 = self.dec4(b, x4)
        # d3 = self.dec3(d4, x3)
        d2 = self.dec2(b, x2)
        d1 = self.dec1(d2, x1)

        # Output
        output = self.output_conv(d1)
        return output

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.float()
            targets = targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


def evaluate_model(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.float()
            targets = targets.float()
       
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_outputs  

def transformArrayToTensor(array):
    return torch.from_numpy(array).float()


data = np.zeros((10,15))
# data[:,5:15] = np.arange(0.1,1.1,0.1)
# data[:,5:15] filled with 1 
data[:,5:15] = 1
plt.imshow(data)
testmask = transformArrayToTensor(data)
testmask = testmask.type(torch.ByteTensor)

target = np.random.uniform(size=(10,15))
target[:,5:15] = target[:,5:15]*np.arange(0.1,1.1,0.1)
targetExt = np.repeat(target[np.newaxis, :, :], 10, axis=0)
targetScaled = targetExt * data
# plt.imshow(targetScaled[0,:,:])

inputs = np.random.uniform(size=(10,15))
inputsExt = np.repeat(inputs[np.newaxis, :, :], 10, axis=0)

'''version where input is scaled with mask'''
inputsScaled = inputsExt * data
inputsScaled = inputsScaled[:, np.newaxis, :, :]
y = targetScaled.copy()
y = y[:, np.newaxis, :, :]
X_m = inputsScaled.copy()

# plt.imshow(X_m[0,0,:,:])
# plt.imshow(y[0,0,:,:])

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_m, y, test_size=0.4, random_state=42)
train_dataset = TensorDataset(transformArrayToTensor(X_train_all), transformArrayToTensor(y_train_all))
test_dataset = TensorDataset(transformArrayToTensor(X_test_all), transformArrayToTensor(y_test_all))
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_M = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_M.parameters(), lr=0.001)
train_model(model_M, train_loader, criterion, optimizer, num_epochs=5)
test_outputs = evaluate_model(model_M, criterion, test_loader)

plt.figure(figsize=(20, 8))
# ensure same colorplot scale through all subplots
vmin = np.min([test_outputs[0,0,:,:], y_test_all[0,0,:,:]])
vmax = np.max([test_outputs[0,0,:,:], y_test_all[0,0,:,:]])

plt.subplot(2, 4, 1)
plt.title('pred 1 - mseloss')
plt.imshow(test_outputs[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 5)
plt.title('true 1')
plt.imshow(y_test_all[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 2)
plt.title('pred 2')
plt.imshow(test_outputs[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 6)
plt.title('true 2')
plt.imshow(y_test_all[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 3)
plt.title('pred 3')
plt.imshow(test_outputs[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 7)
plt.title('true 3')
plt.imshow(y_test_all[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 4)
plt.title('pred 4')
plt.imshow(test_outputs[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 8)
plt.title('true 4')
plt.imshow(y_test_all[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()


'''version where input is kept as is, mask separately'''
data = np.zeros((10,15))
# data[:,5:15] = np.arange(0.1,1.1,0.1)
# data[:,5:15] filled with 1 
data[:,5:15] = 1
# plt.imshow(data)
testmask = transformArrayToTensor(data)
testmask = testmask.type(torch.ByteTensor)

target = np.random.uniform(size=(10,15))
target[:,5:15] = target[:,5:15]*np.arange(0.1,1.1,0.1)
targetExt = np.repeat(target[np.newaxis, :, :], 10, axis=0)
targetScaled = targetExt * data
# plt.imshow(targetScaled[0,:,:])

inputs = np.random.uniform(size=(10,15))
inputsExt = np.repeat(inputs[np.newaxis, :, :], 10, axis=0)

maskExt = np.repeat(data[np.newaxis, :, :], 10, axis=0)

X_n = np.stack([inputsExt, maskExt], axis=1)
y = targetScaled.copy()
y = y[:, np.newaxis, :, :]

# plt.imshow(X_n[0,1,:,:])
# plt.imshow(X_n[0,0,:,:])
# plt.imshow(y[0,0,:,:])

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_n, y, test_size=0.4, random_state=42)

# remove mask from input data train and test
X_train = X_train_all[:, :-1, :, :]
X_test = X_test_all[:, :-1, :, :]
y_train = y_train_all.copy()
y_test = y_test_all.copy()

# mask extracted from train test split
def train_test_mask(data):
    data_mask = data[:, -1, :, :]
    data_mask = data_mask[:, np.newaxis, :, :]
    data_mask_ex = np.repeat(data_mask, data.shape[1]-1, axis=1)
    data_mask_tensor = transformArrayToTensor(data_mask_ex)
    data_mask_binary = data_mask_tensor.type(torch.ByteTensor)
    data_mask_bool = data_mask_binary.bool()
    return data_mask_bool
X_train_mask = train_test_mask(X_train_all)
X_test_mask = train_test_mask(X_test_all)
y_train_mask = train_test_mask(X_train_all)#does it make sense to use the Xtrain mask as target as well? 
y_test_mask = train_test_mask(X_test_all)

#TODO create new dataset and dataloader for combined loss function incl mask
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data, labels, masks, transform=None):
        """
        Args:
            data (torch.Tensor or numpy array): Input data (e.g., images).
            labels (torch.Tensor or numpy array): Corresponding labels for the input data.
            masks (torch.Tensor or numpy array): Masks corresponding to each input data.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., for data augmentation).
        """
        self.data = data
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input data, label, and mask for the given index
        input_data = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        return input_data, label, mask
        
# Create dataset instances
train_dataset = CustomDataset(X_train, y_train, X_train_mask)
test_dataset = CustomDataset(X_test, y_test, y_test_mask)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


model_C = UNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_C.parameters(), lr=0.001)

def train_model_C(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets, mask in train_loader:
            inputs = inputs.float()
            targets = targets.float()
            # mask = mask.float()
            

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs[mask], targets[mask])
     
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model_C(model_C, train_loader, criterion, optimizer, num_epochs=5)
def evaluate_model_C(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    with torch.no_grad():
        for inputs, targets, masks in test_loader:
            inputs = inputs.float()
            targets = targets.float()
            # masks = masks.float()
       
            outputs = model(inputs)
            loss = criterion(outputs[masks], targets[masks])
            # loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_outputs  

test_outputs = evaluate_model_C(model_C, criterion, test_loader)

plt.figure(figsize=(20, 8))
# ensure same colorplot scale through all subplots
vmin = np.min([test_outputs[0,0,:,:], y_test[0,0,:,:]])
vmax = np.max([test_outputs[0,0,:,:], y_test[0,0,:,:]])

plt.subplot(2, 4, 1)
plt.title('pred 1 -mseloss *mask')
plt.imshow(test_outputs[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 5)
plt.title('true 1')
plt.imshow(y_test[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 2)
plt.title('pred 2')
plt.imshow(test_outputs[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 6)
plt.title('true 2')
plt.imshow(y_test[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 3)
plt.title('pred 3')
plt.imshow(test_outputs[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 7)
plt.title('true 3')
plt.imshow(y_test[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 4)
plt.title('pred 4')
plt.imshow(test_outputs[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 8)
plt.title('true 4')
plt.imshow(y_test[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()


'''version where input is kept as is, mask separately, and specific loss function'''
data = np.zeros((10,15))
data[:,5:15] = 1
# plt.imshow(data)

target = np.random.uniform(size=(10,15))
target[:,5:15] = target[:,5:15]*np.arange(0.1,1.1,0.1)
targetExt = np.repeat(target[np.newaxis, :, :], 10, axis=0)
targetScaled = targetExt * data
# plt.imshow(targetScaled[0,:,:])

inputs = np.random.uniform(size=(10,15))
inputsExt = np.repeat(inputs[np.newaxis, :, :], 10, axis=0)
maskExt = np.repeat(data[np.newaxis, :, :], 10, axis=0)

X_n = np.stack([inputsExt, maskExt], axis=1)
y = targetScaled.copy()
y = y[:, np.newaxis, :, :]

# plt.imshow(X_n[0,1,:,:])
# plt.imshow(X_n[0,0,:,:])
# plt.imshow(y[0,0,:,:])

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_n, y, test_size=0.4, random_state=42)

# remove mask from input data train and test
X_train = X_train_all[:, :-1, :, :]
X_test = X_test_all[:, :-1, :, :]
y_train = y_train_all.copy()
y_test = y_test_all.copy()

# plt.imshow(X_train[0,0,:,:])
# plt.imshow(y_train[0,0,:,:])

# mask extracted from train test split
def train_test_mask(data):
    data_mask = data[:, -1, :, :]
    data_mask = data_mask[:, np.newaxis, :, :]
    data_mask_ex = np.repeat(data_mask, data.shape[1]-1, axis=1)
    return data_mask_ex
X_train_mask = train_test_mask(X_train_all)
X_test_mask = train_test_mask(X_test_all)
y_train_mask = train_test_mask(X_train_all)#does it make sense to use the Xtrain mask as target as well? 
y_test_mask = train_test_mask(X_test_all)

# plt.imshow(X_train_mask[0,0,:,:])
# plt.imshow(y_test_mask[0,0,:,:])


#TODO create new dataset and dataloader for combined loss function incl mask
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data, labels, masks, transform=None):
        """
        Args:
            data (torch.Tensor or numpy array): Input data (e.g., images).
            labels (torch.Tensor or numpy array): Corresponding labels for the input data.
            masks (torch.Tensor or numpy array): Masks corresponding to each input data.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., for data augmentation).
        """
        self.data = data
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input data, label, and mask for the given index
        input_data = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        return input_data, label, mask
        
# Create dataset instances
train_dataset = CustomDataset(X_train, y_train, X_train_mask)
test_dataset = CustomDataset(X_test, y_test, y_test_mask)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_C1 = UNet()
# Define the masked loss function
def masked_loss(y_pred, y_true, mask):
    # Compute the loss 
    mse_loss = nn.MSELoss(reduction='none')
    loss = mse_loss(y_pred, y_true)
    masked_loss = (loss * mask).sum()
    print('masked_loss:', masked_loss)
    non_zero_elements = mask.sum()
    print('non_zero_elements:', non_zero_elements)
    mse_loss_val = masked_loss / non_zero_elements
    print('mse_loss_val:', mse_loss_val)
    return mse_loss_val


optimizer = optim.Adam(model_C1.parameters(), lr=0.001)

def train_model_Costum(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets, mask in train_loader:
            inputs = inputs.float()
            targets = targets.float()
            mask = mask.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = masked_loss(outputs, targets, mask)
            # loss = criterion(outputs, targets)
         
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model_Costum(model_C1, train_loader, criterion, optimizer, num_epochs=5)

def evaluate_model_C(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs = inputs.float()
            targets = targets.float()
            mask = mask.float()
       
            outputs = model(inputs)
            loss = masked_loss(outputs, targets, mask)
            # loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_outputs  

test_outputs = evaluate_model_C(model_C1, criterion, test_loader)

plt.figure(figsize=(20, 8))
# ensure same colorplot scale through all subplots
vmin = np.min([test_outputs[0,0,:,:], y_test[0,0,:,:]])
vmax = np.max([test_outputs[0,0,:,:], y_test[0,0,:,:]])

plt.subplot(2, 4, 1)
plt.title('pred 1 masked_loss')
plt.imshow(test_outputs[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 5)
plt.title('true 1')
plt.imshow(y_test[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 2)
plt.title('pred 2')
plt.imshow(test_outputs[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 6)
plt.title('true 2')
plt.imshow(y_test[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 3)
plt.title('pred 3')
plt.imshow(test_outputs[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 7)
plt.title('true 3')
plt.imshow(y_test[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 4)
plt.title('pred 4')
plt.imshow(test_outputs[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 8)
plt.title('true 4')
plt.imshow(y_test[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()



'''version where input is kept as is, mask separately, and specific loss function where we use nanmean'''
data = np.zeros((10,15))
data[:,5:15] = 1
# plt.imshow(data)

target = np.random.uniform(size=(10,15))
target[:,5:15] = target[:,5:15]*np.arange(0.1,1.1,0.1)
targetExt = np.repeat(target[np.newaxis, :, :], 10, axis=0)
targetScaled = targetExt * data
# plt.imshow(targetScaled[0,:,:])

inputs = np.random.uniform(size=(10,15))
inputsExt = np.repeat(inputs[np.newaxis, :, :], 10, axis=0)
maskExt = np.repeat(data[np.newaxis, :, :], 10, axis=0)

X_n = np.stack([inputsExt, maskExt], axis=1)
y = targetScaled.copy()
y = y[:, np.newaxis, :, :]

# plt.imshow(X_n[0,1,:,:])
# plt.imshow(X_n[0,0,:,:])
# plt.imshow(y[0,0,:,:])

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_n, y, test_size=0.4, random_state=42)

# remove mask from input data train and test
X_train = X_train_all[:, :-1, :, :]
X_test = X_test_all[:, :-1, :, :]
y_train = y_train_all.copy()
y_test = y_test_all.copy()

# plt.imshow(X_train[0,0,:,:])
# plt.imshow(y_train[0,0,:,:])

# mask extracted from train test split
def train_test_mask(data):
    data_mask = data[:, -1, :, :]
    data_mask = data_mask[:, np.newaxis, :, :]
    data_mask_ex = np.repeat(data_mask, data.shape[1]-1, axis=1)
    return data_mask_ex
X_train_mask = train_test_mask(X_train_all)
X_test_mask = train_test_mask(X_test_all)
y_train_mask = train_test_mask(X_train_all)#does it make sense to use the Xtrain mask as target as well? 
y_test_mask = train_test_mask(X_test_all)

# plt.imshow(X_train_mask[0,0,:,:])
# plt.imshow(y_test_mask[0,0,:,:])


#TODO create new dataset and dataloader for combined loss function incl mask
from torch.utils.data import Dataset, DataLoader
class CustomDataset(Dataset):
    def __init__(self, data, labels, masks, transform=None):
        """
        Args:
            data (torch.Tensor or numpy array): Input data (e.g., images).
            labels (torch.Tensor or numpy array): Corresponding labels for the input data.
            masks (torch.Tensor or numpy array): Masks corresponding to each input data.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., for data augmentation).
        """
        self.data = data
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get the input data, label, and mask for the given index
        input_data = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        return input_data, label, mask
        
# Create dataset instances
train_dataset = CustomDataset(X_train, y_train, X_train_mask)
test_dataset = CustomDataset(X_test, y_test, y_test_mask)

# Create DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model_C2 = UNet()
# Define the masked loss function
def masked_loss_nan(y_pred, y_true, mask):
    # Compute the loss
    mse_loss = nn.MSELoss(reduction='none')
    #replace 0 with nan values
    masked_nan = torch.where(mask == 0, torch.nan, mask)
    #create boolean tensor of masked_nan
    masked_nan = torch.isnan(masked_nan)
    print(masked_nan)
    loss = mse_loss(y_pred, y_true)
    masked_loss = torch.where(masked_nan, loss, torch.nan)
    print('masked_loss:', masked_loss)
    mse_loss_val = masked_loss.nanmean()
    print('mse_loss_val:', mse_loss_val)
    return mse_loss_val

optimizer = optim.Adam(model_C2.parameters(), lr=0.001)

def train_model_Costum(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, targets, mask in train_loader:
            inputs = inputs.float()
            targets = targets.float()
            mask = mask.float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = masked_loss_nan(outputs, targets, mask)
            # loss = criterion(outputs, targets)
         
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

train_model_Costum(model_C2, train_loader, criterion, optimizer, num_epochs=5)

def evaluate_model_C(model, criterion, test_loader):
    model.eval()
    test_loss = 0.0
    all_outputs = []
    with torch.no_grad():
        for inputs, targets, mask in test_loader:
            inputs = inputs.float()
            targets = targets.float()
            mask = mask.float()
       
            outputs = model(inputs)
            loss = masked_loss_nan(outputs, targets, mask)
            # loss = criterion(outputs, targets)
            test_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
        print(f"Test Loss: {test_loss/len(test_loader):.4f}")
        all_outputs = np.concatenate(all_outputs, axis=0)
        return all_outputs  

test_outputs = evaluate_model_C(model_C2, criterion, test_loader)

plt.figure(figsize=(20, 8))
# ensure same colorplot scale through all subplots
vmin = np.min([test_outputs[0,0,:,:], y_test[0,0,:,:]])
vmax = np.max([test_outputs[0,0,:,:], y_test[0,0,:,:]])

plt.subplot(2, 4, 1)
plt.title('pred 1 - masked_loss_nan')
plt.imshow(test_outputs[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 5)
plt.title('true 1')
plt.imshow(y_test[0,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 2)
plt.title('pred 2')
plt.imshow(test_outputs[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 6)
plt.title('true 2')
plt.imshow(y_test[1,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 3)
plt.title('pred 3')
plt.imshow(test_outputs[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 7)
plt.title('true 3')
plt.imshow(y_test[2,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()

plt.subplot(2, 4, 4)
plt.title('pred 4')
plt.imshow(test_outputs[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
plt.subplot(2, 4, 8)
plt.title('true 4')
plt.imshow(y_test[3,0,:,:], vmin=vmin, vmax=vmax)
plt.colorbar()
