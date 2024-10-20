## 1Residual_block
# Best hyperparameters: {'lr': 0.0006, 'num_layers': 4, 'kernel_size': 2, 'dropout': 0.17, 'step_size': 21, 'gamma': 0.9, 'epochs': 156}

import time
import h5py
import torch
import torch.nn as nn
from torch.nn import init
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import scipy
import scipy.io
import pickle
import functions as fn
import sys
import os
import random
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt

# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)
train_rate = 0.75
val_rate = 0.25

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)  # Chomp layer to ensure causality
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.downsample = weight_norm(nn.Conv1d(n_inputs, n_outputs, 1)) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight, nonlinearity='relu')
        if self.downsample:
            nn.init.kaiming_uniform_(self.downsample.weight, nonlinearity='relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if out.shape[-1] != res.shape[-1]:  # Ensure the dimensions match
            diff = res.shape[-1] - out.shape[-1]
            out = nn.functional.pad(out, (0, diff))
        return (out + res)

class ResidualBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout):
        super(ResidualBlock, self).__init__()
        self.temporal_block1 = TemporalBlock(n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout)

    def forward(self, x):
        out = self.temporal_block1(x)
        return out

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size, dropout, dilation_growth_rate, output_size):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = int(dilation_growth_rate ** i)
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            # Add ResidualBlock with specified parameters
            layers.append(ResidualBlock(in_channels, out_channels, kernel_size, stride=1, 
                                        dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout))
        
        # Add a final convolution layer to project to the desired output dimensionality
        layers.append(nn.Conv1d(num_channels[-1], output_size, kernel_size=1))
            
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def calc_error(pred, target):
    error = np.sqrt(np.sum((pred - target) ** 2))
    step_error = error / pred.shape[0]
    avg_error = step_error / pred.shape[1] / pred.shape[2]
    return avg_error, step_error, error

def calc_nmse(pred, target):
    nmse = np.sum(np.abs((pred - target))**2/np.abs(target)**2) / pred.size
    return nmse

# Example of how to use this model
num_inputs = 100  
num_channels = [100, 100, 100, 100]  # Updated to 4 layers
kernel_size = 2  # Updated kernel size
dropout = 0.17  # Updated dropout rate
output_size = 100  # Define the output size
dilation_growth_rate = 2

model = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout, dilation_growth_rate=dilation_growth_rate, output_size=output_size)

# Training phase
# python Fixed_1.py High VTV_SDWW 16QAM DPA 40 train 200 128
mode = sys.argv[6]

if mode == 'train':
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    training_snr = sys.argv[5]
    epoch = 156  # Updated to 156 epochs
    BATCH_SIZE = int(sys.argv[8])
    
    mat = loadmat('./{}_{}_{}_{}_TCN_training_dataset.mat'.format(mobility, channel_model, modulation_order, scheme))
    Dataset = mat['TCN_Datasets']
    Dataset = Dataset[0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']
    print('Loaded Dataset Inputs: ', X.shape)  
    print('Loaded Dataset Outputs: ', Y.shape)

    # Reshape Input and Label Data
    input_data_Re = X.reshape(-1, 2)
    label_data_Re = Y.reshape(-1, 2)
    print('Reshaped Training Input Dataset: ', input_data_Re.shape)
    print('Reshaped Training Label Dataset: ', label_data_Re.shape)

    # Normalization
    scaler = StandardScaler()
    input_data_sclar = scaler.fit_transform(input_data_Re)
    label_data_sclar = scaler.fit_transform(label_data_Re)

    # Reshape after normalization
    input_data_sclar = input_data_sclar.reshape(X.shape)
    label_data_sclar = label_data_sclar.reshape(Y.shape)
    print('Reshaped Normalized Training Input Dataset: ', input_data_sclar.shape)
    print('Reshaped Normalized Training Label Dataset: ', label_data_sclar.shape)

    # Training and Validation Datasets splits
    nums = X.shape[0]
    train_indices = []
    val_indices = []

    samples_per_snr = int(nums / len(SNR_index))

    for i in range(len(SNR_index)):
        start_idx = i * samples_per_snr
        end_idx = start_idx + samples_per_snr
        block_indices = np.arange(start_idx, end_idx)
        
        np.random.shuffle(block_indices)
        split_index = int(len(block_indices) * train_rate)
        train_indices.extend(block_indices[:split_index])
        val_indices.extend(block_indices[split_index:])

    # Use indices to create training and validation datasets
    Train_X = input_data_sclar[train_indices]
    Train_Y = label_data_sclar[train_indices]
    Val_X = input_data_sclar[val_indices]
    Val_Y = label_data_sclar[val_indices]

    # Convert arrays to PyTorch tensors
    train_input = torch.from_numpy(Train_X).float()
    train_label = torch.from_numpy(Train_Y).float()
    val_input = torch.from_numpy(Val_X).float()
    val_label = torch.from_numpy(Val_Y).float()

    print('Train_X:', train_input.shape)
    print('Train_Y:', train_label.shape)
    print('Val_X:', val_input.shape)
    print('Val_Y:', val_label.shape)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Generate batch dataset
    dataset = data.TensorDataset(train_input, train_label)
    loader = data.DataLoader(dataset=dataset, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)
    
    val_dataset = data.TensorDataset(val_input, val_label)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0006)  # Updated learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=21, gamma=0.9)  # Updated StepLR parameters

    model_path = './{}_{}_{}_{}_TCN_{}.pt'.format(mobility, channel_model, modulation_order, scheme, training_snr)

    training_losses = []
    validation_losses = []
    total_training_time = 0
    best_val_loss = float('inf')
    early_stopping_patience = 10
    early_stopping_counter = 0

    for ep in range(int(epoch)):
        start_time = time.time()
        model.train()
        total_loss = 0
        total_train_error = 0

        for step, (batch_x, batch_y) in enumerate(loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_err, _, _ = calc_error(outputs.cpu().detach().numpy(), batch_y.cpu().detach().numpy())
            total_train_error += avg_err

        avg_train_loss = total_loss / len(loader)
        avg_train_error = total_train_error / len(loader)
        training_losses.append(avg_train_loss)

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                val_loss = criterion(val_outputs, val_y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)

        print(f'Epoch [{ep+1}/{epoch}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping triggered")
                break

        scheduler.step()
        epoch_time = time.time() - start_time
        total_training_time += epoch_time

    print("Training completed. Total time: {:.2f} seconds.".format(total_training_time))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(training_losses, label='Training Loss', color='blue', linestyle='-')
    plt.plot(validation_losses, label='Validation Loss', color='red', linestyle='--')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_plot.png', dpi=300)
    plt.show()

# Testing phase
elif mode == 'test':
    # We are running the testing phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    testing_snr = sys.argv[5]

    if modulation_order == 'QPSK':
        modu_way = 1
    elif modulation_order == '16QAM':
        modu_way = 2
    
    num_ofdm_symbols = 100
    num_subcarriers = 52
    dposition_WCP = [0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,21,22,23,24,25,26,27,28,29,30,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51]
    ppositions_python = [6, 20, 31, 45]

    DSC_IDX = np.array(dposition_WCP)
    ppositions = np.array(ppositions_python)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Initialize and load the TCN model
    model_path = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_{testing_snr}.pt'
    model = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout, dilation_growth_rate=dilation_growth_rate, output_size=output_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    scaler = StandardScaler()

    with torch.no_grad():         
        for n_snr in SNR_index:
            mat = loadmat(f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_testing_dataset_{n_snr}.mat')
            Dataset = mat['TCN_Datasets']
            Dataset = Dataset[0, 0]
            X = Dataset['Test_X']
            Y = Dataset['Test_Y']
            yf_d = Dataset['Y_DataSubCarriers']
            print('Loaded Dataset Inputs: ', X.shape)
            print('Loaded Dataset Outputs: ', Y.shape)
            print('Loaded Testing OFDM Frames: ', yf_d.shape)

            num_symbols_real_imag = yf_d.shape[1]
            hf_DL_TCN = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
            print("Shape of hf_DL_TCN: ", hf_DL_TCN.shape)

            for i in range(yf_d.shape[0]):
                print(f'Processing Frame | {i}')
                
                initial_channel_est = X[i, :, :]
                initial_channel_est = scaler.fit_transform(initial_channel_est.reshape(-1, 2)).reshape(-1, num_subcarriers)

                input_tensor = torch.from_numpy(initial_channel_est).type(torch.FloatTensor).unsqueeze(0)
                input_tensor = input_tensor.to(device)

                output_tensor = model(input_tensor)
                output_data = scaler.inverse_transform(output_tensor.detach().cpu().numpy().reshape(-1, 2)).reshape(output_tensor.shape)

                for j in range(yf_d.shape[1]):
                    hf_out_real = output_data[0, j * 2, :]
                    hf_out_imag = output_data[0, (j * 2) + 1, :]
                    hf_out = hf_out_real + 1j * hf_out_imag
                    hf_DL_TCN[i, j, :] = hf_out

            result_path = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_Results_{n_snr}.pickle'
            dest_name = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_TCN_Results_{n_snr}.mat'
            with open(result_path, 'wb') as f:
                pickle.dump([X, Y, hf_DL_TCN], f)
            scipy.io.savemat(dest_name, {f'{scheme}_TCN_test_x_{n_snr}': X,
                                         f'{scheme}_TCN_test_y_{n_snr}': Y,
                                         f'{scheme}_TCN_predicted_y_{n_snr}': hf_DL_TCN})
            print("Data successfully converted to .mat file")
            os.remove(result_path)

