% Date: Oct 22, 2016
% Author: Yuxuan Zhang

% for xor.csv
%{
data = csvread('xor.csv');
input = data(:,1:size(data,2)-1)'; % each row is a feature, each column is an instance
target = data(:,end)'; % assume the last column is label
target = cat(1,1-target,target); % one-hot coding for label, each row is a class, each column is an instance
%}

% for iris.csv
%{
data = csvread('iris.csv');
input = data(:,1:size(data,2)-3)';
target = data(:,end-2:end)';
%}

% for ministTrn

load('mnistTrn.mat');
input = trn;
target = trnAns;
%}

% number of nodes in input/output layers
input_size = size(input,1);
output_size = size(target,1);
split = [80,10,10]; % 80% train, 10% validation, 10% test
nodeLayers = [input_size,30,output_size];
numEpochs = 30; 
batchSize = 10;
eta = 3;

trans = 'sigmoid';
%trans = 'tanh';
%trans = 'relu';
%trans = 'softmax';

cost = 'quadratic';
%cost = 'cross-entropy';
%cost = 'log';

mu = 0.3; % friction term used in momentum update
lambda = 5; % used for L2 regularization 
%rng(10); % set random seed

[weights, biases] = Expanded_NN(input,target,split,nodeLayers,numEpochs,batchSize,eta,trans,cost,lambda,mu)
