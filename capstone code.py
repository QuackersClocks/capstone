import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Load dataset (Replace with actual business data)
def load_data():
    data = pd.read_csv("shopping_trends.csv")
    X = data.drop(columns=['target']).values  # Features
    y = data['target'].values  # Target variable (e.g., sales, churn rate)
    return X, y

# Preprocess data
def preprocess_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.float32).view(-1, 1), torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    return X_train, X_test, y_train, y_test

# Define PyTorch model
class BusinessAIModel(nn.Module):
    def __init__(self, input_size):
        super(BusinessAIModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.output(x))  # Sigmoid for binary classification
        return x

# Train model
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Evaluate model
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions > 0.5).float()  # Convert probabilities to binary labels
        accuracy = (predictions == y_test).sum().item() / y_test.size(0)
        print(f"Test Accuracy: {accuracy:.4f}")

# Run AI framework
def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(X, y)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = BusinessAIModel(input_size=X_train.shape[1])
    
    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary Cross Entropy for classification
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train and evaluate
    train_model(model, train_loader, criterion, optimizer)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
