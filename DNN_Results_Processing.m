clc;clearvars;close all; warning('off','all');

mobility = 'Low';
ChType = 'VTV_UC';
modu = '16QAM';
scheme = 'TRFI';
testing_samples = 4000;
if(isequal(modu,'QPSK'))
nBitPerSym       = 2; 
elseif (isequal(modu,'16QAM'))
nBitPerSym       = 4; 
elseif (isequal(modu,'64QAM'))
 nBitPerSym       = 6; 
end
M                     = 2 ^ nBitPerSym; % QAM Modulation Order   
Pow                   = mean(abs(qammod(0:(M-1),M)).^2); % Normalization factor for QAM    
load(['./',mobility,'_',ChType,'_',modu,'_simulation_parameters']);
EbN0dB                    = (0:5:40)'; 
nSym                      = 50;
constlen                  = 7;
trellis                   = poly2trellis(constlen,[171 133]);
tbl                       = 34;
scramInit                 = 93;
nDSC                      = 48;
nUSC                      = 52;
dpositions                = [1:6, 8:20, 22:31, 33:45, 47:52].'; 
Interleaver_Rows          = 16;
Interleaver_Columns       = (nBitPerSym * nDSC * nSym) / Interleaver_Rows;
N_SNR                      = size(EbN0dB,1);
Phf                        = zeros(N_SNR,1);
Err_scheme_DNN             = zeros(N_SNR,1);
Ber_scheme_DNN             = zeros(N_SNR,1);


for n_snr = 1:N_SNR 
    
    % Loading Simulation Parameters Results
    load(['./',mobility,'_',ChType,'_',modu,'_testing_simulation_',num2str(EbN0dB(n_snr)),'.mat']);
    % Loading scheme-DNN Results
    load(['./',mobility,'_',ChType,'_',modu,'_',scheme,'_DNN_Results_',num2str(EbN0dB(n_snr)),'.mat']);

     TestY = eval([scheme,'_DNN_test_y_',num2str(EbN0dB(n_snr))]);
     TestY = TestY.';
     TestY = TestY(1:nUSC,:) + 1i*TestY(nUSC+1:2*nUSC,:);
     TestY = reshape(TestY, nUSC, nSym, testing_samples);
     scheme_DNN = eval([scheme,'_DNN_corrected_y_',num2str(EbN0dB(n_snr))]);
     scheme_DNN = scheme_DNN.';
     scheme_DNN = scheme_DNN(1:nUSC,:) + 1i*scheme_DNN(nUSC+1:2*nUSC,:);
     scheme_DNN = reshape(scheme_DNN, nUSC, nSym, testing_samples);
     True_Channels_Structure = True_Channels_Structure(:,:,:);
     for u = 1:size(scheme_DNN,3)

        H_scheme_DNN = scheme_DNN(dpositions,:,u);

        Phf(n_snr) = Phf(n_snr) + mean(sum(abs(True_Channels_Structure(:,:,u)).^2)); 
        Err_scheme_DNN (n_snr) =  Err_scheme_DNN (n_snr) +  mean(sum(abs(H_scheme_DNN - True_Channels_Structure(dpositions,:,u)).^2)); 
        
        % IEEE 802.11p Rx
        Bits_scheme_DNN     = de2bi((qamdemod(sqrt(Pow) * (Received_Symbols_FFT_Structure(dpositions ,:,u) ./ H_scheme_DNN),M))); 
        %Bits_AE_DNN     = de2bi((qamdemod(sqrt(Pow) * (EqualizedS(:,:,u) ),M)));
        Ber_scheme_DNN(n_snr)   = Ber_scheme_DNN(n_snr) + biterr(wlanScramble(vitdec((matintrlv((deintrlv(Bits_scheme_DNN(:),Random_permutation_Vector)).',Interleaver_Columns,16).'),poly2trellis(7,[171 133]),34,'trunc','hard'),93),TX_Bits_Stream_Structure(:,u));
     end
   toc;
end
Phf = Phf ./ testing_samples;
ERR_TRFI_DNN = Err_scheme_DNN ./ (testing_samples * Phf); 
BER_TRFI_DNN = Ber_scheme_DNN/ (testing_samples * nSym * 48 * nBitPerSym);

%% Save the data in a .mat file
save('TRFI_results.mat', 'Phf', 'ERR_TRFI_DNN', 'BER_TRFI_DNN');
%% Assuming you have already calculated Phf, ERR_scheme_DNN, and BER_scheme_DNN

% Plot Phf
%figure;
%plot(EbN0dB, Phf, 'b-o');
%xlabel('Eb/N0 (dB)');
%ylabel('Phf');
%title('Mean Channel Power (Phf) vs Eb/N0');
%grid on;

% Plot NMSE_scheme_DNN
NMSE_TRFI_DNN = ERR_TRFI_DNN ./ Phf; % Calculate NMSE from ERR_scheme_DNN and Phf
figure;
plot(EbN0dB, NMSE_TRFI_DNN, 'r-s');
xlabel('Eb/N0 (dB)');
ylabel('Normalized Mean Squared Error (NMSE)');
title('Normalized MSE of Predicted Channel vs True Channel (NMSE\_scheme\_DNN) vs Eb/N0');
grid on;

% Plot BER_scheme_DNN
figure;
plot(EbN0dB, BER_TRFI_DNN, 'g-d');
xlabel('Eb/N0 (dB)');
ylabel('Bit Error Rate (BER)');
title('Bit Error Rate (BER\_scheme\_DNN) vs Eb/N0');
grid on;
