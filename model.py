import numpy as np
import torch, torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from IPython.display import display, clear_output
from torch.utils.data import DataLoader, TensorDataset

# initialize weights for linear layers using He initialization 
def weights_init(layer_in):
    if isinstance(layer_in, nn.Linear):
        nn.init.kaiming_uniform_(layer_in.weight, nonlinearity='relu')
        if layer_in.bias is not None:
            nn.init.zeros_(layer_in.bias)

# Function to train a neural network
def train_model(D_i, D_k, D_o, name_dataset, train_data_x, train_data_y, test_data_x, test_data_y, n_epoch):
    # Define the model
    model = nn.Sequential(
      nn.Linear(D_i, D_k), nn.ReLU(),
      nn.Linear(D_k, D_k), nn.ReLU(),
      nn.Linear(D_k, D_k), nn.ReLU(),
      nn.Linear(D_k, D_k), nn.ReLU(),
      nn.Linear(D_k, D_o))

    # Loss function, optimizer, and scheduler
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Convert data to tensors and create DataLoader
    x_train = torch.tensor(train_data_x, dtype=torch.float32)
    y_train = torch.tensor(train_data_y, dtype=torch.long)
    data_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=100, shuffle=True)

    x_test = torch.tensor(test_data_x, dtype=torch.float32)
    y_test = torch.tensor(test_data_y, dtype=torch.long)
    

    # Set device and move model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Apply weight initialization
    model.apply(weights_init)

    # Initialize arrays for storing results
    losses_train = np.zeros(n_epoch)
    accuracies_train = np.zeros(n_epoch)
    losses_test = np.zeros(n_epoch)
    accuracies_test = np.zeros(n_epoch)

    # Training loop
    for epoch in range(n_epoch):
        model.train()
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = loss_function(pred, y_batch)
            loss.backward()
            optimizer.step()

        #Evaluate on training set
        model.eval()
        with torch.no_grad():
            pred_train = model(x_train.to(device))
            loss_epoch_train = loss_function(pred_train, y_train.to(device)).item()
            _, pred_classes_train = torch.max(pred_train, 1)
            accuracy_epoch_train = (pred_classes_train == y_train.to(device)).float().mean().item()

        # Evaluate on test set
        loss_epoch_test, accuracy_epoch_test = test_model(model, x_test, y_test)

        # Store results
        losses_train[epoch] = loss_epoch_train
        accuracies_train[epoch] = accuracy_epoch_train
        losses_test[epoch] = loss_epoch_test
        accuracies_test[epoch] = accuracy_epoch_test

        # Display progress
        if epoch % 5 == 0:
            clear_output(wait=True)
            display(f"Dataset: {name_dataset}, Epoch {epoch}: Train loss = {loss_epoch_train:.4f}, Train accuracy = {accuracy_epoch_train:.4f}, Test loss = {loss_epoch_test:.4f}, Test accuracy = {accuracy_epoch_test:.4f}")

    # Return results
    return losses_train[:epoch+1], accuracies_train[:epoch+1], losses_test[:epoch+1], accuracies_test[:epoch+1]


def test_model(model, x_test, y_test):
    model.eval()
    with torch.no_grad():
        device = next(model.parameters()).device
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        
        outputs = model(x_test)
        loss_function = nn.CrossEntropyLoss()
        loss = loss_function(outputs, y_test).item()

        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean().item()

    return loss, accuracy