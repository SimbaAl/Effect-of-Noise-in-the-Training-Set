clc; clearvars; close all; warning('off', 'all');
% Define Simulation parameters
mobility = 'Low';
modu = '16QAM';
ChType = 'VTV_UC';

filename = ['./', mobility, '_', ChType, '_', modu, '_simulation_parameters.mat'];
loadedData = load(filename);

% Load new results
LSTM_Results = load('LSTM_DPA_TA.mat');
% New results
ERR_DPA_TA_LSTM = LSTM_Results.ERR_DPA_TA_LSTM;
BER_DPA_TA_LSTM = LSTM_Results.BER_DPA_TA_LSTM;

% Load new results
%newResults = load('results_script5.mat');
% New results
%ERR_scheme_DNN = newResults.ERR_scheme_DNN;
%BER_scheme_DNN = newResults.BER_scheme_DNN;

% Load new results
TCNResults = load('TCN_Ber_nmse_results.mat');

% New results
Err_scheme_TCN = TCNResults.Err_scheme_TCN;
Ber_scheme_TCN = TCNResults.Ber_scheme_TCN;

% Load new results
newResults = load('TRFI_results.mat');
% New results
ERR_TRFI_DNN = newResults.ERR_TRFI_DNN;
BER_TRFI_DNN = newResults.BER_TRFI_DNN;

% Load new results
newResults = load('STA_results.mat');
% New results
ERR_STA_DNN = newResults.ERR_STA_DNN;
BER_STA_DNN = newResults.BER_STA_DNN;


BER_Ideal = loadedData.BER_Ideal;
BER_Initial = loadedData.BER_Initial;
BER_DPA_TA = loadedData.BER_DPA_TA;
BER_LS = loadedData.BER_LS;
BER_STA = loadedData.BER_STA;
BER_TRFI = loadedData.BER_TRFI;
BER_CDP = loadedData.BER_CDP;

ERR_Initial = loadedData.ERR_Initial;
ERR_LS = loadedData.ERR_LS;
ERR_DPA_TA = loadedData.ERR_DPA_TA;
ERR_TRFI = loadedData.ERR_TRFI;
ERR_CDP = loadedData.ERR_CDP;
ERR_STA = loadedData.ERR_STA;

% Plotting NMSE
figure;
semilogy(ERR_Initial, 'k--o', 'DisplayName', 'DPA', 'LineWidth', 1.5);
hold on;
semilogy(ERR_LS, 'b-p', 'DisplayName', 'LS', 'LineWidth', 1.5);
semilogy(ERR_STA, 'm-v', 'DisplayName', 'STA', 'LineWidth', 1.5);
semilogy(ERR_CDP, 'r-*', 'DisplayName', 'CDP', 'LineWidth', 1.5);
semilogy(ERR_TRFI, 'g-s', 'DisplayName', 'TRFI', 'LineWidth', 1.5);
semilogy(ERR_DPA_TA, 'c-d', 'DisplayName', 'DPA-TA', 'LineWidth', 1.5);
semilogy(ERR_TRFI_DNN, 'color', [0.5 0 0.5], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'TRFI-DNN', 'LineWidth', 1.5);
semilogy(ERR_STA_DNN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 'x', 'DisplayName', 'STA-DNN', 'LineWidth', 1.5);
semilogy(ERR_DPA_TA_LSTM, 'color', [1, 0, 1], 'LineStyle', '-', 'Marker', '+', 'DisplayName', 'LSTM-DPA-TA', 'LineWidth', 1.5);
semilogy(Err_scheme_TCN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 's', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', [1, 0.5, 0], 'DisplayName', 'TCN');
hold off;
xticklabels({'0','5','10','15','20','25','30','35','40'});
xlabel('SNR (dB)');
ylabel('NMSE');
title('NMSE');
legend('Location', 'northeast');
grid on;
ylim([1e-4, 1e2]);

% Plotting BER
figure;
semilogy(BER_Ideal, 'k--o', 'DisplayName', 'Ideal', 'LineWidth', 1.5);
hold on;
semilogy(BER_Initial, 'k-d', 'DisplayName', 'DPA', 'LineWidth', 1.5);
semilogy(BER_LS, 'b-p', 'DisplayName', 'LS', 'LineWidth', 1.5);
semilogy(BER_STA, 'm-v', 'DisplayName', 'STA', 'LineWidth', 1.5);
semilogy(BER_CDP, 'r-*', 'DisplayName', 'CDP', 'LineWidth', 1.5);
semilogy(BER_TRFI, 'g-s', 'DisplayName', 'TRFI', 'LineWidth', 1.5);
semilogy(BER_DPA_TA, 'c-d', 'DisplayName', 'DPA-TA', 'LineWidth', 1.5);
semilogy(BER_TRFI_DNN, 'color', [0.5 0 0.5], 'LineStyle', '-', 'Marker', '^', 'DisplayName', 'TRFI-DNN', 'LineWidth', 1.5);
semilogy(BER_STA_DNN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 'x', 'DisplayName', 'STA-DNN', 'LineWidth', 1.5);
semilogy(BER_DPA_TA_LSTM, 'color', [1, 0, 1], 'LineStyle', '-', 'Marker', '+', 'DisplayName', 'LSTM-DPA-TA', 'LineWidth', 1.5);
semilogy(Ber_scheme_TCN, 'color', [1, 0.5, 0], 'LineStyle', '-', 'Marker', 's', 'LineWidth', 2.5, 'MarkerSize', 8, 'MarkerFaceColor', [1, 0.5, 0], 'DisplayName', 'TCN');
hold off;
xticklabels({'0','5','10','15','20','25','30','35','40'});
xlabel('SNR (dB)');
ylabel('BER');
title('BER');
legend('Location', 'southwest');
grid on;
ylim([1e-5, 1e0]);