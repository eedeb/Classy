import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split  # Added for validation split

from nltk_utils import bag_of_words, tokenize, stem
#nltk.download('punkt_tab')
from model import NeuralNet

# Load JSON file
with open('intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Process and prepare the data
all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Stem, lower and clean punctuation
ignore_words = ['?', '.', '!', '"']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# Create training data
X = []
y = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X.append(bag)
    label = tags.index(tag)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=None)

# Define hyperparameters
num_epochs = 300  # Reduced due to early convergence
batch_size = 64
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)

print(f"{input_size} > {hidden_size} > {output_size}")

# Define dataset class
class ChatDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.n_samples = len(x_data)
        self.x_data = x_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Create data loaders for training and validation
train_dataset = ChatDataset(X_train, y_train)
val_dataset = ChatDataset(X_val, y_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Select device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Define loss and optimizer (with regularization via weight decay)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop
best_val_loss = float('inf')
early_stop_counter = 0
patience = 10  # Stop if no improvement for 10 evaluations

for epoch in range(num_epochs):
    model.train()
    for words, labels in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Validation step
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for words, labels in val_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)

    # Logging
    if (epoch + 1) % 25 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

print(f'Final training loss: {loss.item():.4f}')
print(f'Final validation loss: {val_loss:.4f}')

# Save trained model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "hidden_size_2": 0,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. Model saved to {FILE}")

# Basic model evaluation (accuracy on validation set)
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for words, labels in val_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Validation Accuracy: {100 * correct / total:.2f}%')
