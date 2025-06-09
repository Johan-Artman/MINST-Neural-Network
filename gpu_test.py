import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets import mnist
import numpy as np
import time
from tqdm import tqdm
from collections import Counter

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST data
print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print data distribution
print("\nTraining data distribution:")
train_dist = Counter(y_train)
for digit in range(10):
    print(f"Digit {digit}: {train_dist[digit]} samples")

print("\nTest data distribution:")
test_dist = Counter(y_test)
for digit in range(10):
    print(f"Digit {digit}: {test_dist[digit]} samples")

# Convert to PyTorch tensors and normalize
# Reshape to (batch_size, channels, height, width)
x_train = torch.FloatTensor(x_train).reshape(-1, 1, 28, 28) / 255.0
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test).reshape(-1, 1, 28, 28) / 255.0
y_test = torch.LongTensor(y_test)

# Print shapes for debugging
print(f"\nTraining data shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# Move data to GPU if available
x_train = x_train.to(device)
y_train = y_train.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

# Create a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(5 * 28 * 28, 100)  # Adjusted for padding
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        # Print shape for debugging
        print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        print(f"After conv shape: {x.shape}")
        x = self.sigmoid(x)
        x = x.view(-1, 5 * 28 * 28)  # Adjusted for padding
        print(f"After reshape shape: {x.shape}")
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Create model and move to GPU
model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training function
def train_epoch(model, train_loader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # Create progress bar for batches with mininterval
    pbar = tqdm(train_loader, desc='Training batches', leave=False, mininterval=0.5)
    
    for batch_idx, (data, target) in enumerate(pbar):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Update progress bar with current metrics, but less frequently
        if batch_idx % 10 == 0:  # Update every 10 batches
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
        
    return total_loss / len(train_loader), 100. * correct / total

# Create data loaders with larger batch size
train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # Increased batch size

# Training loop
print("\nStarting training...")
start_time = time.time()

# Create progress bar for epochs with mininterval
epochs = 20
epoch_pbar = tqdm(range(epochs), desc='Training epochs', mininterval=1.0)

for epoch in epoch_pbar:
    loss, accuracy = train_epoch(model, train_loader, criterion, optimizer)
    epoch_pbar.set_postfix({
        'loss': f'{loss:.4f}',
        'acc': f'{accuracy:.2f}%'
    })

end_time = time.time()
print(f"\nTraining completed in {end_time - start_time:.2f} seconds")

# Save the model
torch.save(model.state_dict(), 'gpu_trained_model.pth')
print("Model saved to gpu_trained_model.pth")

# Test the model
print("\nEvaluating model on test set...")
model.eval()
with torch.no_grad():
    # Create progress bar for test set with mininterval
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=128)  # Increased batch size
    correct = 0
    total = 0
    digit_correct = [0] * 10
    digit_total = [0] * 10
    
    for data, target in tqdm(test_loader, desc='Testing', mininterval=0.5):
        output = model(data)
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Calculate per-digit accuracy
        for i in range(10):
            mask = (target == i)
            digit_correct[i] += pred[mask].eq(target[mask]).sum().item()
            digit_total[i] += mask.sum().item()
    
    accuracy = 100. * correct / total
    print(f"\nOverall Test Accuracy: {accuracy:.2f}%")
    print("\nPer-digit accuracy:")
    for i in range(10):
        if digit_total[i] > 0:
            digit_acc = 100. * digit_correct[i] / digit_total[i]
            print(f"Digit {i}: {digit_acc:.2f}%") 