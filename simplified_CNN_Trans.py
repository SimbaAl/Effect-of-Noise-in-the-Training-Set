##subcarrier as sequence length

import time
import h5py
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import scipy
import scipy.io
import sys
import os
import random
import functions as fn
import pickle
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from copy import deepcopy
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.cuda.amp import GradScaler, autocast

# General Parameters
configuration_mode = len(sys.argv)
Train_SNR = np.arange(0, 45, 5)
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

'''class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=64):  # Set max_len to 128 or 256
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x'''

# Rotary Positional Encoding for Transformer
class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len

    def forward(self, x):
        seq_len = x.size(1)
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return x * emb.cos() + self.rotate_half(x) * emb.sin()

    def rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_layers, num_heads, hidden_dim, dropout, output_size):
        super(TransformerModel, self).__init__()
        
        # Two CNN layers
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Transformer components
        self.input_linear = nn.Linear(hidden_dim, hidden_dim)
        self.pos_encoder = RotaryPositionalEncoding(hidden_dim)
        encoder_layers = TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.output_linear = nn.Linear(hidden_dim, output_size)

    def forward(self, src):
        # CNN layers
        src = src.permute(0, 2, 1)
        src = self.conv1(src)
        src = self.bn1(src)
        src = self.relu(src)
        src = self.dropout(src)
        
        src = self.conv2(src)
        src = self.bn2(src)
        src = self.relu(src)
        src = self.dropout(src)
        
        # Prepare for Transformer
        src = src.permute(2, 0, 1)  # (seq_len, batch_size, hidden_dim)
        
        # Transformer layers
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_linear(output.permute(1, 0, 2))
        
        return output

def calc_error(pred, target):
    error = np.sqrt(np.sum((pred - target) ** 2))
    step_error = error / pred.shape[0]
    avg_error = step_error / pred.shape[1] / pred.shape[2]
    return avg_error, step_error, error

def calc_nmse(pred, target):
    nmse = np.sum(np.abs((pred - target))**2/np.abs(target)**2) / pred.size
    return nmse

# Example of how to use this simplified model
input_size = 100
num_layers = 2  # Reduced from 4
num_heads = 2   # Reduced from 4
hidden_dim = 128 # Reduced from 128
dropout = 0.1
output_size = 100

model = TransformerModel(input_size, num_layers, num_heads, hidden_dim, dropout, output_size)

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
    epoch = int(sys.argv[7])
    BATCH_SIZE = int(sys.argv[8])

    # Set a random seed for reproducibility
    np.random.seed(42)

    Train_SNR = np.arange(0, 45, 5)
    train_rate = 0.75

    # Load data
    mat = loadmat('./{}_{}_{}_{}_Transformer_training_dataset.mat'.format(mobility, channel_model, modulation_order, scheme))
    Dataset = mat['Transformer_Datasets'][0, 0]
    X = Dataset['Train_X']
    Y = Dataset['Train_Y']

    print('Loaded Dataset Inputs: ', X.shape)  
    print('Loaded Dataset Outputs: ', Y.shape)

    # Split data
    nums = X.shape[0]
    samples_per_snr = int(nums / len(Train_SNR))

    train_indices = []
    val_indices = []

    for i in range(len(Train_SNR)):
        start_idx = i * samples_per_snr
        end_idx = start_idx + samples_per_snr
        indices = np.arange(start_idx, end_idx)
        np.random.shuffle(indices)
        split_idx = int(samples_per_snr * train_rate)
        train_indices.extend(indices[:split_idx])
        val_indices.extend(indices[split_idx:])

    # Split the data
    Train_X, Train_Y = X[train_indices], Y[train_indices]
    Val_X, Val_Y = X[val_indices], Y[val_indices]

    # Normalization (fit only on training data)
    scaler = StandardScaler()
    Train_X_flat = Train_X.reshape(-1, 2)
    scaler.fit(Train_X_flat)

    # Transform both training and validation data
    Train_X_norm = scaler.transform(Train_X_flat).reshape(Train_X.shape)
    Val_X_norm = scaler.transform(Val_X.reshape(-1, 2)).reshape(Val_X.shape)

    # Normalize Y data
    scaler_y = StandardScaler()
    Train_Y_flat = Train_Y.reshape(-1, 2)
    scaler_y.fit(Train_Y_flat)

    Train_Y_norm = scaler_y.transform(Train_Y_flat).reshape(Train_Y.shape)
    Val_Y_norm = scaler_y.transform(Val_Y.reshape(-1, 2)).reshape(Val_Y.shape)

    # Transpose the data to have the desired sequence length
    Train_X = np.transpose(Train_X_norm, (0, 2, 1))
    Train_Y = np.transpose(Train_Y_norm, (0, 2, 1))
    Val_X = np.transpose(Val_X_norm, (0, 2, 1))
    Val_Y = np.transpose(Val_Y_norm, (0, 2, 1))

    # Convert to PyTorch tensors
    train_input = torch.from_numpy(Train_X).float()
    train_label = torch.from_numpy(Train_Y).float()
    val_input = torch.from_numpy(Val_X).float()
    val_label = torch.from_numpy(Val_Y).float()

    print('Train_X:', train_input.shape)
    print('Train_Y:', train_label.shape)
    print('Val_X:', val_input.shape)
    print('Val_Y:', val_label.shape)

    
    '''# Assuming mobility, channel_model, modulation_order, scheme are already defined
    mat = loadmat('./{}_{}_{}_{}_Transformer_training_dataset.mat'.format(mobility, channel_model, modulation_order, scheme))
    Dataset = mat['Transformer_Datasets']
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
    input_data_sclar = scaler.fit_transform(input_data_Re)  # .reshape(input_data.shape)
    label_data_sclar = scaler.fit_transform(label_data_Re)  # .reshape(label_data.shape)

    # Reshape after normalization
    input_data_sclar = input_data_sclar.reshape(X.shape)
    label_data_sclar = label_data_sclar.reshape(Y.shape)
    print('Reshaped Normalized Training Input Dataset: ', input_data_sclar.shape)
    print('Reshaped Normalized Training Label Dataset: ', label_data_sclar.shape)

    # Transpose the data to have the desired sequence length
    Train_X = np.transpose(input_data_sclar, (0, 2, 1))
    Train_Y = np.transpose(label_data_sclar, (0, 2, 1))
    Val_X = np.transpose(input_data_sclar, (0, 2, 1))
    Val_Y = np.transpose(label_data_sclar, (0, 2, 1))

    # Training and Validation Datasets splits
    nums = X.shape[0]
    train_indices = []
    val_indices = []
    samples_per_snr = int(nums / len(Train_SNR))

    for i in range(len(Train_SNR)):
        start_idx = i * samples_per_snr
        end_idx = start_idx + samples_per_snr
        block_indices = np.arange(start_idx, end_idx)
        np.random.shuffle(block_indices)
        split_index = int(len(block_indices) * train_rate)
        train_indices.extend(block_indices[:split_index])
        val_indices.extend(block_indices[split_index:])

    Train_X = Train_X[train_indices]
    Train_Y = Train_Y[train_indices]
    Val_X = Val_X[val_indices]
    Val_Y = Val_Y[val_indices]

    train_input = torch.from_numpy(Train_X).float()
    train_label = torch.from_numpy(Train_Y).float()
    val_input = torch.from_numpy(Val_X).float()
    val_label = torch.from_numpy(Val_Y).float()

    print('Train_X:', train_input.shape)
    print('Train_Y:', train_label.shape)
    print('Val_X:', val_input.shape)
    print('Val_Y:', val_label.shape)'''
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    dataset = data.TensorDataset(train_input, train_label)
    loader = data.DataLoader(dataset=dataset, batch_size=int(BATCH_SIZE), shuffle=True, num_workers=2 if torch.cuda.is_available() else 0)
    
    val_dataset = data.TensorDataset(val_input, val_label)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.89)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    model_path = './{}_{}_{}_{}_Transformer_{}.pt'.format(mobility, channel_model, modulation_order, scheme, training_snr)

    training_losses = []
    validation_losses = []



    '''def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, model_save_path=model_path):
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        early_stopping_patience = 10
        total_training_time = 0
        scaler = GradScaler()

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)
            start_time = time.time()

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                num_batches_trained = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        with autocast():
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)

                        if phase == 'train':
                            scaler.scale(loss).backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                            scaler.step(optimizer)
                            scaler.update()

                    running_loss += loss.item() * inputs.size(0)
                    num_batches_trained += 1

                epoch_loss = running_loss / num_batches_trained

                if phase == "train":
                    training_losses.append(epoch_loss)
                else:
                    validation_losses.append(epoch_loss)

                print(f"{phase} Loss: {epoch_loss:.4f}")

                if phase == "val":
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_state = deepcopy(model.state_dict())
                        early_stopping_counter = 0
                        torch.save(best_model_state, model_save_path)
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= early_stopping_patience:
                            print("Early stopping triggered")
                            return model

            scheduler.step(validation_losses[-1])  # Assuming you switch to ReduceLROnPlateau
            epoch_time = time.time() - start_time
            total_training_time += epoch_time

        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Best model saved with loss: {best_val_loss:.4f} at {model_save_path}")

        print("Training completed. Total time: {:.2f} seconds.".format(total_training_time))
        
        return model'''


    def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25, model_save_path=model_path):
        best_val_loss = float('inf')
        best_model_state = None
        early_stopping_counter = 0
        early_stopping_patience = 10
        total_training_time = 0

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            print("-" * 10)
            start_time = time.time()

            for phase in ["train", "val"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                num_batches_trained = 0

                for inputs, labels in dataloaders[phase]:
                    #print(f'Inputs shape: {inputs.shape}')
                    #print(f'Labels shape: {labels.shape}')
                    
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        #print(f'Outputs shape: {outputs.shape}')
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    num_batches_trained += 1

                epoch_loss = running_loss / num_batches_trained

                if phase == "train":
                    training_losses.append(epoch_loss)
                else:
                    validation_losses.append(epoch_loss)

                print(f"{phase} Loss: {epoch_loss:.4f}")

                if phase == "val":
                    if epoch_loss < best_val_loss:
                        best_val_loss = epoch_loss
                        best_model_state = deepcopy(model.state_dict())
                        early_stopping_counter = 0
                        torch.save(best_model_state, model_save_path)
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter >= early_stopping_patience:
                            print("Early stopping triggered")
                            break

            scheduler.step()
            epoch_time = time.time() - start_time
            total_training_time += epoch_time

            if early_stopping_counter >= early_stopping_patience:
                break

        if best_model_state:
            model.load_state_dict(best_model_state)
            print(f"Best model saved with loss: {best_val_loss:.4f} at {model_save_path}")

        print("Training completed. Total time: {:.2f} seconds.".format(total_training_time))
        
        return model

    train_loader = DataLoader(TensorDataset(train_input, train_label), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_input, val_label), batch_size=BATCH_SIZE, shuffle=False)
    dataloaders = {"train": train_loader, "val": val_loader}

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=epoch)
    
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
# python W_A.py High VTV_SDWW 16QAM DPA 15 test
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

    # Initialize and load the Transformer model
    model_path = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_Transformer_{testing_snr}.pt'
    #input_size = 100  
    #num_layers = 4
    #num_heads = 4
    #hidden_dim = 100
    #dropout = 0.0013
    #output_size = 100  

    #model = ImprovedTransformerModel(input_size, num_layers, num_heads, hidden_dim, dropout, output_size)
    model = TransformerModel(input_size, num_layers, num_heads, hidden_dim, dropout, output_size)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    scaler = StandardScaler()
    
    with torch.no_grad():         
        for n_snr in SNR_index:
            mat = loadmat(f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_Transformer_testing_dataset_{n_snr}.mat')
            Dataset = mat['Transformer_Datasets']
            Dataset = Dataset[0, 0]
            X = Dataset['Test_X']
            Y = Dataset['Test_Y']
            yf_d = Dataset['Y_DataSubCarriers']
            print('Loaded Dataset Inputs: ', X.shape)
            print('Loaded Dataset Outputs: ', Y.shape)
            print('Loaded Testing OFDM Frames: ', yf_d.shape)

            num_symbols_real_imag = yf_d.shape[1]
            hf_DL_Transformer = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
            print("Shape of hf_DL_Transformer: ", hf_DL_Transformer.shape)

            for i in range(yf_d.shape[0]):
                print(f'Processing Frame | {i}')
                
                initial_channel_est = X[i, :, :]
                initial_channel_est = scaler.fit_transform(initial_channel_est.reshape(-1, 2)).reshape(-1, num_subcarriers)

                input_tensor = torch.from_numpy(initial_channel_est).type(torch.FloatTensor).unsqueeze(0)
                input_tensor = input_tensor.permute(0, 2, 1).to(device)  # Transpose to (batch_size, seq_len, input_size)

                output_tensor = model(input_tensor)

                output_data = scaler.inverse_transform(output_tensor.detach().cpu().numpy().reshape(-1, 2)).reshape(output_tensor.shape)

                for j in range(yf_d.shape[1]):
                    
                    hf_out_real = output_data[0, :, j * 2]
                    hf_out_imag = output_data[0, :, (j * 2) + 1]
                    hf_out = hf_out_real + 1j * hf_out_imag
                    
                    hf_DL_Transformer[i, j, :] = hf_out

                    #if j == 0:
                    hf_DL_Transformer[i, j, :] = hf_out
                    #else:
                    '''y_eq = yf_d[i ,j, :] / hf_out
                    
                    #q = fn.map(fn.demap(y_eq, modu_way), modu_way)
                    
                    sf = yf_d[i, j, :] / hf_out
                    x = fn.demap(sf, modu_way)
                    xf = fn.map(x, modu_way)
                    
                    hf_DL_Transformer[i, j, :] = yf_d[i, j, :] / xf
                    hf_out  = yf_d[i, j, :] / xf
                    #hf_DL_Transformer[i, j, :] = q
                    
                    if j < yf_d.shape[1] - 1:
                        next_sym_real = np.real(hf_DL_Transformer[i, j, :])
                        next_sym_imag = np.imag(hf_DL_Transformer[i, j, :])
                        next_sym_index = 2 * j
                        # Update the channel estimate in X for the next OFDM symbol
                        #initial_channel_est[next_sym_index, DSC_IDX] = hf_out.real
                        #initial_channel_est[next_sym_index + 1, DSC_IDX] = hf_out.imag
                        
                        #X[i, 2 * (j + 1), DSC_IDX] = next_sym_real
                        #X[i, 2 * (j + 1) + 1, DSC_IDX] = next_sym_imag
                        #initial_channel_est = X[i, 2 * (j + 1), dpositions_array]
                        #initial_channel_est = X[i, 2 * (j + 1) + 1, dpositions_array]
                        
                        # Combine with the previous estimate if necessary
                        # (Here, assuming a simple averaging approach)
                        #X[i, j + 1, :] = 0.5 * X[i, j, :] + 0.5 * next_sym_real
                        #X[i, yf_d.shape[1] + j + 1, :] = 0.5 * X[i, yf_d.shape[1] + j, :] + 0.5 * next_sym_imag

                        X[i, next_sym_index,:] = hf_out.real
                        X[i, next_sym_index + 1,:] = hf_out.imag'''

            result_path = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_Transformer_Results_{n_snr}.pickle'
            dest_name = f'./{mobility}_{channel_model}_{modulation_order}_{scheme}_Transformer_Results_{n_snr}.mat'
            with open(result_path, 'wb') as f:
                pickle.dump([X, Y, hf_DL_Transformer], f)
            scipy.io.savemat(dest_name, {f'{scheme}_Transformer_test_x_{n_snr}': X,
                                         f'{scheme}_Transformer_test_y_{n_snr}': Y,
                                         f'{scheme}_Transformer_predicted_y_{n_snr}': hf_DL_Transformer})
            print("Data successfully converted to .mat file")
            os.remove(result_path)

##Input Data Shape:
#The original input shape is (18000, 100, 52), where:
#18000 is the number of samples
#100 is the number of symbols (time steps)
#52 is the number of subcarriers (features)

##Data Reshaping:
#After reshaping and normalization, the data is transposed to (13500, 52, 100) for training, where:
#13500 is the number of training samples
#52 is now the number of subcarriers (features)
#100 is still the number of symbols (time steps)


#CNN Part:
#In the CNN part of the model, the input is first permuted to (batch_size, 100, 52):
#Here, 100 becomes the number of input channels, and 52 becomes the sequence length for the CNN.
#The convolutions are applied along the last dimension (52), treating it as the sequence length or time steps for the CNN part.

#Transformer Part:
#Before feeding the data into the Transformer, it's permuted again:
#This changes the shape to (52, batch_size, 100), where:
#52 becomes the sequence length for the Transformer
#100 is now the feature dimension
#In the Transformer architecture, the first dimension is typically used as the sequence length or time steps.

#To summarize:
#In the CNN part, the sequence length (or time steps) is 52 (the number of subcarriers).
#In the Transformer part, the sequence length (or time steps) is also 52.
#The model essentially treats the subcarriers as the sequence dimension, applying convolutions across them in the CNN part, and then using them as the sequence input for the Transformer. The original time dimension (100 symbols) is treated as channels in the CNN and then as features in the Transformer.


#how is this model doing this:
#This architecture allows the model to capture both local patterns within each subcarrier (using CNNs) and long-range dependencies across subcarriers (using the Transformer).

##Capturing local patterns within each subcarrier (using CNNs):

#After the initial permutation, the data shape is (batch_size, 100, 52), where 100 is the number of input channels (originally the number of symbols), and 52 is the sequence length (number of subcarriers).
#The CNN layers (conv1, conv2, conv3) apply 1D convolutions along the last dimension (52 subcarriers). Each convolution operation looks at a small window of adjacent subcarriers at a time (kernel size of 3).
#This allows the CNN to detect local patterns and features within small groups of adjacent subcarriers. For example, it might detect edges, peaks, or other local structures in the frequency domain.
#The use of multiple CNN layers allows the model to build up increasingly complex local features.


##Capturing long-range dependencies across subcarriers (using the Transformer):

#After the CNN layers, the data is permuted to (52, batch_size, 100) for the Transformer.
#In this arrangement, each of the 52 subcarriers is treated as a "token" or time step in the sequence for the Transformer.
#The Transformer's self-attention mechanism allows each subcarrier to attend to all other subcarriers, regardless of their position in the sequence. This captures long-range dependencies and relationships across the entire range of subcarriers.
#The multi-head attention allows the model to focus on different types of relationships between subcarriers in parallel.
#The positional encoding added by the PositionalEncoding layer helps the Transformer understand the original order of the subcarriers, which might be important for some patterns.


##Combining local and global information:

#By first using CNNs and then feeding the result into a Transformer, the model can leverage both types of information:
#a) The local features extracted by the CNNs
#b) The global relationships captured by the Transformer
#This combination allows the model to make predictions based on both fine-grained local structures and broader, long-range patterns across the frequency domain.


#In the context of your wireless communication application, this architecture might help in:

#Detecting local frequency-domain effects like fading or interference (via CNNs)
#Understanding broader spectral patterns and interdependencies across the entire bandwidth (via Transformer)
#Combining this information to make more accurate channel estimations or signal predictions
