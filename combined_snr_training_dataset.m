clc; clearvars; close all; warning('off', 'all');

% Load pre-defined DNN Testing Indices
load('./samples_indices_8000.mat');

configuration = 'training'; % training or testing
mobility = 'Low';
modu = '16QAM';
ChType = 'VTV_UC';
scheme = 'DPA_TA';

nSym = 50; % Number of OFDM symbols
nSC_In = 104; % Input subcarriers (52 real + 52 imag)
nSC_Out = 104; % Output subcarriers (52 real + 52 imag) - Adjusted to match nSC_In for consistency

% Define whether to load training or testing data
if isequal(configuration, 'training')
    indices = training_samples;
    EbN0dB = 0:5:40;
else
    indices = testing_samples;
    EbN0dB = 0:5:40;
end

Dataset_size = size(indices,1);
SNR = EbN0dB.';
N_SNR = length(SNR);

% Load the combined dataset
load(['./', mobility, '_', ChType, '_', modu, '_', configuration, '_simulation_combined.mat'], ...
    'Combined_True_Channels_Structure', ['Combined_' scheme '_Structure'], 'Combined_HLS_Structure', 'Combined_R_Symbols_Training_Structure');

% Initialize the Train_X and Train_Y arrays to store concatenated data across SNRs
Train_X = zeros(nSC_In/2, 2 * nSym, Dataset_size * N_SNR);
Train_Y = zeros(nSC_Out/2, 2 * nSym, Dataset_size * N_SNR);

current_index = 1;

for n_snr = 1:N_SNR
    Dataset_X = zeros(nSC_In/2, 2*nSym, Dataset_size);
    Dataset_Y = zeros(nSC_Out/2, 2*nSym, Dataset_size);

    temp_structure = eval(['Combined_' scheme '_Structure']);
    scheme_Channels_Structure = temp_structure(:,:,:,n_snr);
    True_Channels_Structure = Combined_True_Channels_Structure(:,:,:,n_snr);

    for i = 1:Dataset_size
        for sym = 1:nSym
            Train_X(:, 2 * sym - 1, current_index + i - 1) = real(scheme_Channels_Structure(:, sym, i));
            Train_X(:, 2 * sym, current_index + i - 1) = imag(scheme_Channels_Structure(:, sym, i));
            Train_Y(:, 2 * sym - 1, current_index + i - 1) = real(True_Channels_Structure(:, sym, i));
            Train_Y(:, 2 * sym, current_index + i - 1) = imag(True_Channels_Structure(:, sym, i));
        end
    end

    current_index = current_index + Dataset_size;  % Update the index for the next SNR level
end

% Permute dimensions [3, 2, 1] for Train_X and Train_Y
Train_X = permute(Train_X, [3, 2, 1]);
Train_Y = permute(Train_Y, [3, 2, 1]);

% Organize data into a structure
TCN_Datasets.Train_X = Train_X;
TCN_Datasets.Train_Y = Train_Y;

% Save the consolidated dataset
save(['./', mobility, '_', ChType, '_', modu, '_', scheme, '_TCN_', configuration, '_dataset.mat'], 'TCN_Datasets', '-v7.3');

%save(['./', mobility, '_', ChType, '_', modu, '_', scheme, '_TCN_', configuration, '_dataset.mat'], 'TCN_Datasets');
