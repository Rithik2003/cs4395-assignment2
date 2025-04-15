import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define the neural network model
class FFNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Define the dataset class
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000)  # Limiting to 1000 features
        
        # Fit the vectorizer on the training data and transform the text to features
        texts = [item["text"] for item in data]
        self.features = self.vectorizer.fit_transform(texts).toarray()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get the features and label (star rating)
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(item["stars"], dtype=torch.float32)
        
        return {"features": features, "label": label}

# Load data from json files
def load_data(train_data_path, val_data_path):
    with open(train_data_path) as training_f:
        train_data = json.load(training_f)
    with open(val_data_path) as validation_f:
        valid_data = json.load(validation_f)
    return train_data, valid_data

# Training function
def train(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        features = batch["features"]
        labels = batch["label"]
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs.squeeze(), labels)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    return avg_loss

# Validation function
def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            features = batch["features"]
            labels = batch["label"]
            
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            
            running_loss += loss.item()
    
    avg_loss = running_loss / len(val_loader)
    return avg_loss

# Main function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_dim', type=int, required=True, help='Number of hidden units in the network')
    parser.add_argument('--epochs', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data JSON file')
    parser.add_argument('--val_data', type=str, required=True, help='Path to the validation data JSON file')
    parser.add_argument('--do_train', action='store_true', help='Flag to indicate training')
    args = parser.parse_args()

    # Load the data
    train_data, valid_data = load_data(args.train_data, args.val_data)

    # Create datasets and dataloaders
    train_dataset = ReviewDataset(train_data)
    valid_dataset = ReviewDataset(valid_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Define model, loss function, and optimizer
    input_dim = 1000  # Number of features from the TF-IDF vectorizer
    output_dim = 1    # Single output (rating)
    
    model = FFNN(input_dim, args.hidden_dim, output_dim)
    criterion = nn.MSELoss()  # Using MSE for regression task (star ratings)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    if args.do_train:
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            # Train the model
            train_loss = train(model, train_loader, criterion, optimizer)
            print(f"Training Loss: {train_loss:.4f}")
            
            # Validate the model
            val_loss = validate(model, val_loader, criterion)
            print(f"Validation Loss: {val_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "ffnn_model.pth")
    print("Model saved to ffnn_model.pth")

if __name__ == '__main__':
    main()
