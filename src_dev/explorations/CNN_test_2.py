### Step 1: Import Required Libraries

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from sklearn.model_selection import train_test_split
import torch.nn.functional as F  # Importing the functional module
import matplotlib.pyplot as plt
import numpy as np


### Step 2: Define the Dataset Class

# Assume `precipitation_maps` is a NumPy array of shape `(N, H, W)` where `N` is the number of samples, `H` is the height, and `W` is the width of each precipitation map. The `groundwater_levels` is a NumPy array of shape `(N,)` with the corresponding groundwater levels.

class GroundwaterDataset(Dataset):
    def __init__(self, precipitation_maps, groundwater_levels, transform=None):
        self.precipitation_maps = precipitation_maps
        self.groundwater_levels = groundwater_levels
        self.transform = transform
    def __len__(self):
        return len(self.precipitation_maps)
    def __getitem__(self, idx):
        image = self.precipitation_maps[idx]
        label = self.groundwater_levels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

### Step 3: Define the CNN Model

# Here's a simple CNN model with a few convolutional layers followed by fully connected layers.

class GroundwaterCNN(nn.Module):
    def __init__(self):
        super(GroundwaterCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)  # Adjust input size based on input image size
        self.fc2 = nn.Linear(128, 1)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten the tensor
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

### Step 4: Prepare Data and Create DataLoader

# Example data (replace with actual data)
precipitation_maps = np.random.rand(1000, 64, 64)  # 1000 samples of 64x64 images
groundwater_levels = np.random.rand(1000)  # 1000 corresponding labels

# Reshape precipitation maps to add channel dimension (1 channel for grayscale images)
precipitation_maps = precipitation_maps[:, np.newaxis, :, :]  # Now shape is (1000, 1, 64, 64)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(precipitation_maps, groundwater_levels, test_size=0.2, random_state=42)

# Transformations
transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32))  # Convert to torch tensor
])

# Create datasets
train_dataset = GroundwaterDataset(X_train, y_train, transform=transform)
test_dataset = GroundwaterDataset(X_test, y_test, transform=transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


### Step 5: Train the Model

# Initialize model, loss function, and optimizer
model = GroundwaterCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

### Step 6: Evaluate the Model

model.eval()
test_loss = 0.0
predictions = []
actuals = []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        test_loss += loss.item()
        
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(labels.tolist())

print(f"Test Loss: {test_loss/len(test_loader)}")

### Step 7: Plot the results scatter

def plot_predictions_vs_actuals(predictions, actuals, num_samples=50):
    plt.figure(figsize=(15, 7))
    indices = np.arange(len(predictions[:num_samples]))
    plt.plot(indices, predictions[:num_samples], 'r', label='Predicted', marker='o')
    plt.plot(indices, actuals[:num_samples], 'b', label='Actual', marker='x')
    plt.xlabel('Sample index')
    plt.ylabel('Groundwater Level')
    plt.title('Predicted vs Actual Groundwater Levels')
    plt.legend()
    plt.show()

# Call the plotting function
plot_predictions_vs_actuals(predictions, actuals, num_samples=50)



### Step 8: Evaluate the Model spatially
map_dataset = GroundwaterDataset(precipitation_maps, groundwater_levels, transform=transform)
map_loader = DataLoader(map_dataset, batch_size=32, shuffle=False)

model.eval()
test_loss = 0.0
predictions = []
actuals = []
with torch.no_grad():
    for inputs, labels in map_loader:
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        test_loss += loss.item()
        
        predictions.extend(outputs.squeeze().tolist())
        actuals.extend(labels.tolist())

print(f"Test Loss: {test_loss/len(test_loader)}")



### Step 9: Plot the results as a map

def plot_predictions_vs_actuals(predictions, actuals):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Predictions vs Actuals', fontsize=16)
    # Reshape the images for visualization
    pred_image = np.reshape(predictions, (20, 50))
    actual_image = np.reshape(actuals, (20, 50))
    axes[0].imshow(pred_image, cmap='viridis')
    axes[0].set_title(f'Predicted: {np.mean(predictions):.4f}')
    axes[0].axis('off')
    axes[1].imshow(actual_image, cmap='viridis')
    axes[1].set_title(f'Actual: {np.mean(actuals):.4f}')
    axes[1].axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

plot_predictions_vs_actuals(predictions, actuals)