% This script is to launch the training of the neural network, based on the
% training data.
% =========================================================================

clear variables;
close all;

% Load training data and essential parameters
load('trainData.mat','XTrain','YTrain');

numSC = 64;

% Batch size
miniBatchSize = 4000;

% Iteration
maxEpochs = 30;

% Sturcture
inputSize = 2*numSC*3;
numHiddenUnits = 128; 
numHiddenUnits2 = 64;
numHiddenUnits3 = numSC;
numClasses = 16;

% DNN Layers
layers = [ ...
    sequenceInputLayer(inputSize)

    fullyConnectedLayer(numHiddenUnits)
    reluLayer('Name','relu1')

    fullyConnectedLayer(numHiddenUnits2)
    reluLayer('Name','relu2')

    fullyConnectedLayer(numClasses)
    sigmoidLayer('Name','sig1')

    %classificationLayer
    ];

% Training options
options = trainingOptions('adam',...
    'InitialLearnRate',0.01,...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'LearnRateDropFactor',0.1,...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Verbose',1,...
    'Plots','training-progress'); 

% Train the neural network
tic;
net = trainNetwork(XTrain,YTrain,layers,options);
toc;

save('NN1.mat','net');

