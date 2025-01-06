import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Assume you have a dataset class
class GroundwaterDataset(torch.utils.data.Dataset):
    # Your dataset class for loading spatial input and groundwater heads
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        x = self.inputs[idx]
        y = self.targets[idx]
        return x, y

# UNet architecture (use the previously defined one)
class UNet(nn.Module):
    # Define your U-Net structure here
    pass

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * inputs.size(0)
        
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        print(f"Training Loss: {epoch_train_loss:.4f}")
        
        # Validation Phase
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)
        
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        print(f"Validation Loss: {epoch_val_loss:.4f}")
        
        # Save best model based on validation loss
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")

def test_model(model, test_loader, criterion):
    model.eval()
    running_test_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item() * inputs.size(0)
    
    test_loss = running_test_loss / len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

# Dataset and DataLoader setup
inputs = ...  # Load your spatial input data
targets = ...  # Load your target data (groundwater heads)

dataset = GroundwaterDataset(inputs, targets)

# Split the dataset into train, validation, and test sets
train_size = int(0.7 * len(dataset))  # 70% for training
val_size = int(0.15 * len(dataset))   # 15% for validation
test_size = len(dataset) - train_size - val_size  # 15% for testing

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders for each dataset
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, criterion, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.MSELoss()  # For regression tasks like groundwater heads prediction
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training and validation
num_epochs = 10
train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# Load best model for testing
model.load_state_dict(torch.load("best_model.pth"))

# Testing
test_model(model, test_loader, criterion)
