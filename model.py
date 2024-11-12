# Nikhil Patil
# CSEC 620 Project 3
# Phishing Neural Network Model

# Imports
import random, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# Decide Device to be used depending on machine ran on
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class PhishingDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature, label

class PhishingNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PhishingNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

data = pd.read_csv('dataset_full.csv')

X = data.drop(columns=['phishing']).values
y = data['phishing'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,100)) # random for reproducibility

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create datasets and loaders
train_dataset = PhishingDataset(X_train, y_train)
test_dataset = PhishingDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

input_size = X_train.shape[1]
hidden_size = 128
model = PhishingNN(input_size, hidden_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# train model
num_epochs = 15
time_start = time.perf_counter()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

time_end = time.perf_counter()

all_labels = []
all_preds = []

# evaluate model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, targets in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1) # discard first return value
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

        all_labels.extend(targets.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())


accuracy = 100 * correct / total
time_elapsed = time_end - time_start
conf_matrix = confusion_matrix(all_labels, all_preds)
class_report = classification_report(all_labels, all_preds, target_names=["Not Phishing", "Phishing"])

print(f"Time Taken: {time_elapsed:.2f} seconds")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Classification Report: {class_report}")
sns.heatmap(conf_matrix, annot=True, fmt=".2f",cmap="Blues")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
plt.savefig('confusion_matrix.png')
plt.show()
