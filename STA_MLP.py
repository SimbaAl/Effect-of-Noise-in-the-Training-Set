import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
import scipy
import scipy.io
import pickle
from scipy.io import loadmat
import functions as fn
import os
from copy import deepcopy
from fastai.basics import *
import sys
import os
import h5py

# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)

# Define the model in PyTorch
class DNN(nn.Module):
    def __init__(self, input_size, hidden_layer1, hidden_layer2, output_size):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_layer1)
        self.layer2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.layer3 = nn.Linear(hidden_layer2, output_size)

        # Initialize weights
        nn.init.trunc_normal_(self.layer1.weight, mean=0.0, std=0.05)
        nn.init.trunc_normal_(self.layer2.weight, mean=0.0, std=0.05)
        nn.init.trunc_normal_(self.layer3.weight, mean=0.0, std=0.05)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.layer3(x)
        return x

# Training phase
if configuration_mode == 9:
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    training_snr = sys.argv[5]
    dnn_input = int(sys.argv[6])
    dnn_output = int(sys.argv[7])
    batch_size = int(sys.argv[8])

    # Best hyperparameters
    best_lr = 0.001
    num_layers = 2
    hidden_layer_sizes = [29, 27]
    epochs = 133


    # Assuming mobility, channel_model, modulation_order, scheme are already defined
    filename = './{}_{}_{}_{}_DNN_training_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, training_snr)
    #filename = './{}_{}_{}_{}_TCN_training_dataset.mat'.format(mobility, channel_model, modulation_order, scheme)
    with h5py.File(filename, 'r') as file:
        Dataset = file['DNN_Datasets']
        X = np.array(Dataset['Train_X'])
        Y = np.array(Dataset['Train_Y'])

    print('Loaded Dataset Inputs: ', X.shape)
    print('Loaded Dataset Outputs: ', Y.shape)
    
    # Transpose the datasets to bring subcarriers to the second dimension
    X = np.transpose(X, (1, 0))
    Y = np.transpose(Y, (1, 0))

    # Transpose the datasets to bring subcarriers to the second dimension
    print('Transposed_X: ', X.shape)
    print('Transposed_Y: ', Y.shape)    

    # Assuming mobility, channel_model, modulation_order, scheme are already defined
    '''mat = loadmat('./{}_{}_{}_{}_DNN_training_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, training_snr))
    Dataset = mat['DNN_Datasets']
    Dataset = Dataset[0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']
    print('Loaded Dataset Inputs: ', X.shape)  # Size: Training_Samples x 2Kon
    print('Loaded Dataset Outputs: ', Y.shape)  # Size: Training_Samples x 2Kon'''

    # Normalizing Datasets
    scalerx = StandardScaler()
    scalerx.fit(X)
    scalery = StandardScaler()
    scalery.fit(Y)
    scalerx_path = './{}_{}_{}_{}_scalerx.pkl'.format(mobility, channel_model, modulation_order, scheme, training_snr)
    scalery_path = './{}_{}_{}_{}_scalery.pkl'.format(mobility, channel_model, modulation_order, scheme, training_snr)

    # After fitting the scalers on the training data
    with open(scalerx_path, 'wb') as f:
        pickle.dump(scalerx, f)

    with open(scalery_path, 'wb') as f:
        pickle.dump(scalery, f)

    XS = scalerx.transform(X)
    YS = scalery.transform(Y)

    # Convert numpy arrays to PyTorch tensors
    XS = torch.from_numpy(XS).float()
    YS = torch.from_numpy(YS).float()

    # Split the dataset into training and validation sets
    XS_train, XS_val, YS_train, YS_val = train_test_split(XS, YS, test_size=0.25)

    # Create TensorDatasets for training and validation
    train_dataset = TensorDataset(XS_train, YS_train)
    val_dataset = TensorDataset(XS_val, YS_val)

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the model with best hyperparameters
    model = DNN(dnn_input, hidden_layer_sizes[0], hidden_layer_sizes[1], dnn_output)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=best_lr)

    # Reduce learning rate on plateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.01, patience=20, min_lr=0.000001)

    # Variables for early stopping
    early_stopping_patience = 20
    early_stopping_counter = 0
    best_loss = float('inf')

    model_path = './{}_{}_{}_{}_DNN_{}.pt'.format(mobility, channel_model, modulation_order, scheme, training_snr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation step
        model.eval()
        val_total_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                val_total_loss += val_loss.item()
                val_batches += 1

        val_loss = val_total_loss / val_batches

        print(f"Epoch {epoch + 1}, Training Loss: {avg_loss}, Validation Loss: {val_loss}")

        # Checkpoint saving
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), model_path)
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        # Reduce LR on plateau
        scheduler.step(val_loss)

# Testing phase
else:
    # We are running the testing phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    testing_snr = sys.argv[5]
    dnn_input = 104
    hidden_layer1 = 29
    hidden_layer2 = 27
    dnn_output = 104
    model = DNN(dnn_input, hidden_layer1, hidden_layer2, dnn_output)
    model_path = './{}_{}_{}_{}_DNN_{}.pt'.format(mobility, channel_model, modulation_order, scheme, testing_snr)
    model.load_state_dict(torch.load(model_path))
    scalerx_path = './{}_{}_{}_{}_scalerx.pkl'.format(mobility, channel_model, modulation_order, scheme, testing_snr)
    scalery_path = './{}_{}_{}_{}_scalery.pkl'.format(mobility, channel_model, modulation_order, scheme, testing_snr)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        for j in SNR_index:
            mat = loadmat('./{}_{}_{}_{}_DNN_testing_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, j))
            Dataset = mat['DNN_Datasets']
            Dataset = Dataset[0, 0]
            X = Dataset['Test_X']
            Y = Dataset['Test_Y']
            print('Loaded Dataset Inputs: ', X.shape)
            print('Loaded Dataset Outputs: ', Y.shape)

            with open(scalerx_path, 'rb') as f:
                scalerx = pickle.load(f)

            with open(scalery_path, 'rb') as f:
                scalery = pickle.load(f)

            XS = scalerx.transform(X)
            YS = scalery.transform(Y)

            XS = torch.from_numpy(XS).float()
            YS = torch.from_numpy(YS).float()

            # Predict
            Y_pred = model(XS).cpu().numpy()  # Move to CPU if using GPU
            Original_Testing_X = scalerx.inverse_transform(XS)
            Original_Testing_Y = scalery.inverse_transform(YS)
            Prediction_Y = scalery.inverse_transform(Y_pred)

            result_path = './{}_{}_{}_{}_DNN_Results_{}.pickle'.format(mobility, channel_model, modulation_order, scheme, j)
            dest_name = './{}_{}_{}_{}_DNN_Results_{}.mat'.format(mobility, channel_model, modulation_order, scheme, j)
            with open(result_path, 'wb') as f:
                pickle.dump([Original_Testing_X, Original_Testing_Y, Prediction_Y], f)

            a = pickle.load(open(result_path, "rb"))
            scipy.io.savemat(dest_name, {
                '{}_DNN_test_x_{}'.format(scheme, j): a[0],
                '{}_DNN_test_y_{}'.format(scheme, j): a[1],
                '{}_DNN_corrected_y_{}'.format(scheme, j): a[2]
            })
            print("Data successfully converted to .mat file ")
            os.remove(result_path)


