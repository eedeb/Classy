import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from nltk_utils import embed_sentence, embed_sentences
from model import NeuralNet

# Reproducibility: a fixed seed makes the train/val split and weight init
# deterministic, so retrains are comparable run to run.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Load JSON file
with open(r'train\agent_intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

# Process and prepare the data
tags = []
xy = []

# Stem, lower and clean punctuation (must match classify() in the inference
# path so the embeddings the model sees at train and run time line up).
ignore_words = ['?', '.', '!', '"']


def clean(text):
    for i in ignore_words:
        text = text.replace(i, "")
    return text.lower()


for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        xy.append((clean(pattern), tag))

# Remove duplicates and sort
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)

# Create training data. Embed all patterns in one batched call — far faster
# than encoding sentence by sentence.
patterns = [p for (p, _) in xy]
X = np.array(embed_sentences(patterns))
y = np.array([tags.index(tag) for (_, tag) in xy])

# Stratified split keeps every class proportionally represented in the
# validation set — important given the class sizes range ~7x.
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Define hyperparameters
num_epochs = 300
batch_size = 32
learning_rate = 0.001
input_size = len(X_train[0])
hidden_size = 64
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

# Inverse-frequency class weights counter the imbalance: rarer intents
# (e.g. Greeting, Banter) get a larger loss contribution so they are not
# drowned out by big classes like Search and Edit.
class_counts = np.bincount(y_train, minlength=output_size).astype(np.float64)
class_weights = class_counts.sum() / (output_size * np.maximum(class_counts, 1))
class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Training loop. We keep the best-on-validation weights rather than whatever
# the loop happens to end on, and restore them before saving.
best_val_loss = float('inf')
best_state = None
early_stop_counter = 0
patience = 20  # Stop if no improvement for 20 evaluations

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
        correct, total = 0, 0
        for words, labels in val_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            outputs = model(words)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

    if (epoch + 1) % 1 == 0 or epoch == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Early stopping check — track and keep the best validation state.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

# Restore the best-performing weights before saving/evaluating.
if best_state is not None:
    model.load_state_dict(best_state)

print(f'Best validation loss: {best_val_loss:.4f}')

# Save trained model. Schema is unchanged so the FreeClaw / basic.py loader
# keeps working: input_size, hidden_size, output_size, tags, model_state.
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)
print(f"Training complete. Model saved to {FILE}")

# Evaluation: overall accuracy plus a per-class breakdown so weak intents
# are visible instead of hidden behind the majority classes.
model.eval()
correct = 0
total = 0
per_class_correct = np.zeros(output_size, dtype=np.int64)
per_class_total = np.zeros(output_size, dtype=np.int64)
with torch.no_grad():
    for words, labels in val_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = model(words)
        _, predicted = torch.max(outputs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for lbl, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
            per_class_total[lbl] += 1
            if lbl == pred:
                per_class_correct[lbl] += 1

print(f'\nValidation Accuracy: {100 * correct / total:.2f}%')
print("Per-class validation accuracy:")
for i, tag in enumerate(tags):
    if per_class_total[i]:
        acc = 100 * per_class_correct[i] / per_class_total[i]
        print(f"  {tag:<20} {acc:6.2f}%  ({per_class_correct[i]}/{per_class_total[i]})")
