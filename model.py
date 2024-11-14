# Nikhil Patil
# CSEC 620 Project 3
# Phishing Neural Network Model

# Imports
import random, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    """
    Simple Class to load the phishing dataset.
    """

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
    """
    Class to initialize the neural network to be used in this assignment
    """

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


def load_data():
    """
    Name: load_data \n
    Purpose: The purpose of this function is to load the data and create the train test split
    :return: X_train, X_test, y_train, y_test, the dataset split via sci-kit learn train test split
    """
    data = pd.read_csv('dataset_full.csv')

    X = data.drop(columns=['phishing']).values
    y = data['phishing'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random.randint(1,
                                                                                                         100))  # random for reproducibility
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def create_dataset_and_dataloader():
    """
    Name: create_dataset_and_dataloader \n
    Purpose: The purpose of this function is to create datasets and create dataloaders based on those datasets. It then returns them so that they can be used to train the model.
    :return: train_dataset, test_dataset, train_loader, test_loader - these are datasets and loaders to be used when creating the model
    """
    X_train, X_test, y_train, y_test = load_data()
    train_dataset = PhishingDataset(X_train, y_train)
    test_dataset = PhishingDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_dataset, test_dataset, train_loader, test_loader


def train_model():
    """
    Name: train_model \n
    Purpose: The purpose of this function is to train the model.
    :return: model - the neural network model, time_elapsed - so the amount of time the model took can be recorded and be used for performance evaluation
    """
    X_train, X_test, y_train, y_test = load_data()
    train_dataset, test_dataset, train_loader, test_loader = create_dataset_and_dataloader()

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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

    time_end = time.perf_counter()

    time_elapsed = time_end - time_start

    return model, time_elapsed


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Name: plot_confusion_matrix \n
    Purpose: The purpose of this function is to plot a confusion matrix.
    :param y_true: y-axis true data (TRUE OUTCOME)
    :param y_pred: y-axis predicted data (Predicted outcome)
    :param title: Title of the Confusion Matrix
    :return: Null
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Not Phishing", "Phishing"],
                yticklabels=["Not Phishing", "Phishing"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.savefig(title)
    plt.show()


def evaluate_model():
    """
    Name: evaluate_model \n
    Purpose: Evaluate the model and its performance.
    :return: Null
    """
    train_dataset, test_dataset, train_loader, test_loader = create_dataset_and_dataloader()
    model, time_elapsed = train_model()
    all_labels = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)  # discard first return value
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    class_report = classification_report(all_labels, all_preds, target_names=["Not Phishing", "Phishing"])

    print(f"Time Taken: {time_elapsed:.2f} seconds")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Classification Report:\n{class_report}")
    plot_confusion_matrix(all_labels, all_preds, title="Neural Network Confusion Matrix")


def random_forest_application():
    """
    Name: random_forest_application \n
    Purpose: Apply the random forest machine learning algorithm to see how it performs on this dataset in comparison to a traditional deep learning neural network and K-nearest neighbors
    :return: Null
    """
    X_train, X_test, y_train, y_test = load_data()
    rf_model = RandomForestClassifier(n_estimators=100, random_state=random.randint(1, 100))
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_preds, target_names=["Not Phishing", "Phishing"]))
    plot_confusion_matrix(y_test, rf_preds, title="Random Forest Confusion Matrix")


def K_nearest_neighbors_application():
    """
    Name: K_nearest_neighbors_application \n
    Purpose: Apply the K-nearest neighbor machine learning algorithm to see how it performs on this dataset in comparison to a traditional deep learning network and the Random Forest classifier.
    :return:
    """
    X_train, X_test, y_train, y_test = load_data()
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_preds = knn_model.predict(X_test)

    print("\nK-Nearest Neighbors Classification Report:")
    print(classification_report(y_test, knn_preds, target_names=["Not Phishing", "Phishing"]))
    plot_confusion_matrix(y_test, knn_preds, title="K-nearest neighbors Confusion Matrix")


def main():
    evaluate_model()
    print("\n")
    random_forest_application()
    print("\n")
    K_nearest_neighbors_application()


if __name__ == '__main__':
    main()
