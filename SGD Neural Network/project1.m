% Date: Oct 9, 2016
% Author: Yuxuan Zhang

% for xor.csv
data = csvread('xor.csv');
train = data(:,1:size(data,2)-1)';
label = data(:,end)'; % assume the last column is label
label = cat(1,1-label,label); % one-hot coding for label


% for iris.csv
%{
data = csvread('iris.csv');
train = data(:,1:size(data,2)-3)';
label = data(:,end-2:end)';
%}

% for ministTrn
%{
load('mnistTrn.mat');
train = trn;
label = trnAns;
%}

% number of nodes in input/output layers
input_size = size(train,1);
output_size = size(label,1);

nodeLayers = [input_size,30,output_size];
numEpochs = 30;
batchSize = 10;
eta = 3;

NN(train, label,nodeLayers,numEpochs,batchSize,eta)

    
