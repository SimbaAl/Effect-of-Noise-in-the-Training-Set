import time
import copy
import sys
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import scipy
import pickle
from scipy.io import loadmat
import functions as fn
import os
from copy import deepcopy

class LSTM(nn.Module):
    def __init__(self, input_size, lstm_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.lstm_size = lstm_size
        self.lstmcell = nn.LSTMCell(input_size=self.input_size, hidden_size=self.lstm_size)
        self.out = nn.Sequential(nn.Linear(lstm_size, 96))

    def forward(self, x_cur, h_cur=None, c_cur=None):
        batch_size, _ = x_cur.size()
        if h_cur is None and c_cur is None:
            h_cur = torch.zeros(batch_size, self.lstm_size, device=x_cur.device)
            c_cur = torch.zeros(batch_size, self.lstm_size, device=x_cur.device)
        h_next, c_next = self.lstmcell(x_cur, (h_cur, c_cur))
        out = self.out(h_next)
        return out, h_next, c_next


def calc_error(pred, target):
    error = np.sqrt(np.sum((pred - target) ** 2))
    step_error = error / pred.shape[0]
    avg_error = step_error / pred.shape[1] / pred.shape[2]
    return avg_error, step_error, error


def calc_nmse(pred, target):
    nmse = np.sum(np.abs((pred - target))**2/np.abs(target)**2) / pred.size
    return nmse


# General Parameters
configuration_mode = len(sys.argv)
SNR_index = np.arange(0, 45, 5)
train_rate = 0.75
val_rate = 0.25

if configuration_mode == 10:
    # We are running the training phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3] 
    scheme = sys.argv[4]
    training_snr = sys.argv[5]
    input_size = int(sys.argv[6])
    lstm_size = 128  # Updated LSTM size
    EPOCH = 160  # Updated training epochs
    BATCH_SIZE = int(sys.argv[9])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load dataset
    filename = './{}_{}_{}_{}_LSTM_training_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, training_snr)
    with h5py.File(filename, 'r') as file:
        Dataset = file['LSTM_Datasets']
        X = np.array(Dataset['Train_X'])
        Y = np.array(Dataset['Train_Y'])

    print('Loaded Dataset Inputs: ', X.shape)
    print('Loaded Dataset Outputs: ', Y.shape)

    # Transpose the datasets to bring subcarriers to the second dimension
    X = np.transpose(X, (2, 1, 0))
    Y = np.transpose(Y, (2, 1, 0))
    print('Transposed_X: ', X.shape)
    print('Transposed_Y: ', Y.shape)

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
    
    # Sequential train-test split
    nums_per_snr = int(X.shape[0] / len(SNR_index))
    train_indices, val_indices = [], []

    for i in range(len(SNR_index)):
        start_idx = i * nums_per_snr
        end_idx = start_idx + nums_per_snr
        block_indices = np.arange(start_idx, end_idx)
        np.random.shuffle(block_indices)

        split_index = int(len(block_indices) * train_rate)
        train_indices.extend(block_indices[:split_index])
        val_indices.extend(block_indices[split_index:])

    XS_train = input_data_sclar[train_indices]
    YS_train = label_data_sclar[train_indices]
    XS_val = input_data_sclar[val_indices]
    YS_val = label_data_sclar[val_indices]
    print('Train_X :', XS_train.shape)
    print('Train_Y :', YS_train.shape)
    print('Val_X :', XS_val.shape)
    print('Val_Y :', YS_val.shape)

    # Convert to PyTorch Tensors
    train_input = torch.from_numpy(XS_train).type(torch.FloatTensor)
    train_label = torch.from_numpy(YS_train).type(torch.FloatTensor)
    val_input = torch.from_numpy(XS_val).type(torch.FloatTensor)
    val_label = torch.from_numpy(YS_val).type(torch.FloatTensor)

    # Create DataLoaders for training and validation
    train_dataset = TensorDataset(train_input, train_label)
    val_dataset = TensorDataset(val_input, val_label)
    loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Train the LSTM model
    model = LSTM(input_size, lstm_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.004)  # Updated learning rate
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=35, gamma=0.7)  # Updated StepLR parameters
    criterion = nn.MSELoss()

    LOSS_TRAIN = []
    LOSS_VAL = []
    min_err = float('inf')
    best_model_wts = deepcopy(model.state_dict())

    for epoch in range(EPOCH):
        # Training
        model.train()
        scheduler.step()
        for step, (train_batch, label_batch) in enumerate(loader):
            train_batch, label_batch = train_batch.to(device), label_batch.to(device)
            optimizer.zero_grad()
            output = torch.zeros_like(label_batch)
            for t in range(train_batch.size(1)):
                if t == 0:
                    out_t, hn, cn = model(train_batch[:, t, :])
                else:
                    out_t, hn, cn = model(train_batch[:, t, :], hn, cn)
                output[:, t, :] = out_t
            loss = criterion(output, label_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1e-4)
            optimizer.step()
            avg_err, _, _ = calc_error(output.detach().cpu().numpy(), label_batch.detach().cpu().numpy())
            if step % 200 == 0:
                print('Epoch: ', epoch, '| Step: ', step, '| loss: ', loss.item(), '| err: ', avg_err)
                LOSS_TRAIN.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_input, val_label = val_input.to(device), val_label.to(device)
            output = torch.zeros_like(val_label)
            for t in range(val_input.size(1)):
                if t == 0:
                    val_t, hn, cn = model(val_input[:, t, :])
                else:
                    val_t, hn, cn = model(val_input[:, t, :], hn, cn)
                output[:, t, :] = val_t
            loss = criterion(output, val_label)
            avg_err, _, _ = calc_error(output.detach().cpu().numpy(), val_label.detach().cpu().numpy())
            print('Epoch: ', epoch, '| val err: ', avg_err)
            LOSS_VAL.append(loss.item())

            if avg_err < min_err:
                min_err = avg_err
                best_model_wts = deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    torch.save(model.to('cpu'), './{}_{}_{}_{}_LSTM_{}.pkl'.format(mobility, channel_model, modulation_order, scheme, training_snr))

else:
    # Testing phase
    mobility = sys.argv[1]
    channel_model = sys.argv[2]
    modulation_order = sys.argv[3]
    scheme = sys.argv[4]
    testing_snr = sys.argv[5]

    for n_snr in SNR_index:
        mat = loadmat('./{}_{}_{}_{}_LSTM_testing_dataset_{}.mat'.format(mobility, channel_model, modulation_order, scheme, n_snr))
        Dataset = mat['LSTM_Datasets']
        Dataset = Dataset[0, 0]
        X = Dataset['Test_X']
        Y = Dataset['Test_Y']
        yf_d = Dataset['Y_DataSubCarriers']
        print('Loaded Dataset Inputs: ', X.shape)
        print('Loaded Dataset Outputs: ', Y.shape)
        print('Loaded Testing OFDM Frames: ', yf_d.shape)
        hf_DL = np.zeros((yf_d.shape[0], yf_d.shape[1], yf_d.shape[2]), dtype="complex64")
        device = torch.device("cpu")
        NET = torch.load('./{}_{}_{}_{}_LSTM_{}.pkl'.format(mobility, channel_model, modulation_order, scheme, testing_snr)).to(device)
        scaler = StandardScaler()

        # Testing
        with torch.no_grad():
            for i in range(yf_d.shape[0]):
                hf = X[i, 0, :]
                hn, cn = None, None
                print('Testing Frame | ', i)
                for j in range(yf_d.shape[1]):
                    hf_input = hf
                    input1 = scaler.fit_transform(hf_input.reshape(-1, 2)).reshape(hf_input.shape)
                    input2 = torch.from_numpy(input1).type(torch.FloatTensor).unsqueeze(0)
                    output, hn, cn = NET(input2.to(device), hn, cn)
                    out = scaler.inverse_transform(output.detach().cpu().numpy().reshape(-1, 2)).reshape(output.shape)
                    hf_out = out[:, :48] + 1j * out[:, 48:]
                    hf_DL[i, j, :] = hf_out
                    sf = yf_d[i, j, :] / hf_out
                    x = fn.demap(sf, modu_way)
                    xf = fn.map(x, modu_way)
                    hf_out = yf_d[i, j, :] / xf
                    hf_out = hf_out.ravel()
                    if j < yf_d.shape[1] - 1:
                        hf_out_Expanded = np.concatenate((hf_out.real, hf_out.imag), axis=0)
                        X[i, j + 1, DSC_IDX] = hf_out_Expanded
                        hf = 0.5 * hf + 0.5 * X[i, j + 1, :]
        
        # Save Results
        result_path = './{}_{}_{}_{}_LSTM_Results_{}.pickle'.format(mobility, channel_model, modulation_order, scheme, n_snr)
        dest_name = './{}_{}_{}_{}_LSTM_Results_{}.mat'.format(mobility, channel_model, modulation_order, scheme, n_snr)
        with open(result_path, 'wb') as f:
            pickle.dump([X, Y, hf_DL], f)

        a = pickle.load(open(result_path, "rb"))
        scipy.io.savemat(dest_name, {
            '{}_test_x_{}'.format(scheme, n_snr): a[0],
            '{}_test_y_{}'.format(scheme, n_snr): a[1],
            '{}_corrected_y_{}'.format(scheme, n_snr): a[2]
        })
        print("Data successfully converted to .mat file ")
        os.remove(result_path)

