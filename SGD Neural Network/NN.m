
% Date: Oct 9, 2016
% Author: Yuxuan Zhang

function NN(train,label,nodeLayers,numEpochs,batchSize,eta)

    % placeholder for weights between different layers
    % the first elem is w between first and second layers, so there
    % will be length(nodeLayers)-1 columns
    weights = cell(1,length(nodeLayers)-1);

    % placeholder for biases between different layers
    biases = cell(1,length(nodeLayers)-1);

    % placeholder for weighted sum
    % the first elem is weighted sum between layer 1 and 2
    z = cell(1,length(nodeLayers)-1);

    % placeholder for activation nodes
    % the second elem are hidden nodes in second layer
    a = cell(1,length(nodeLayers));

    % placeholder for gradient of loss w.r.t. activation function for all
    % layers except first layer
    % so if we have 4-layered network, it then contains 3 elements
    delta_layers = cell(1,length(nodeLayers)-1);

    % placeholder for gradient of loss w.r.t. weights/biases, for all connects between
    % layers
    delta_weights = cell(1,length(nodeLayers)-1);
    delta_biases = cell(1,length(nodeLayers)-1);

    % initialize weights and biases between layers
    for i=1:length(nodeLayers)-1 % for each layer starting from the second layer
        weights{i} = randn(nodeLayers(i+1),nodeLayers(i)); % sampling from normal distribution
        biases{i} = zeros(nodeLayers(i+1),batchSize);
    end

    % loop through epochs
    for epoch=1:numEpochs
        % shuffle data
        cat_data = cat(1,train,label);
        rand_order = randperm(size(cat_data,2)); 
        cat_data = cat_data(:,rand_order); 

        % divide data by batchSize and loop through each mini-batch 
        for index = 1:batchSize:size(cat_data,2)
            minibatch = cat_data(:,index:index+batchSize-1);
            mini_train = minibatch(1:nodeLayers(1),:);
            mini_label = minibatch(nodeLayers(1)+1:end,:);

            % the first elem are input nodes
            a{1} = mini_train; 

            % feed-forward
            for conn=1:length(nodeLayers)-1 % for each connections between layers
                z{conn} = weights{conn}*a{conn} + biases{conn}; % weighted sum
                a{conn+1} = logsig(z{conn}); % sigmoid transformation of z
            end

            % back-propagation

            % gradient of loss (delat) w.r.t. activation function in layers
            % except the first layer.
            % If we have 3-layered network, then we compute delta for layer2 and
            % output layer
            delta_layers{length(nodeLayers)} = (a{end}-mini_label) .* ((1-a{end}).* a{end}); % for output layer
            for layer=length(nodeLayers)-1:-1:2 % going backwards
                delta_layers{layer} = (weights{layer}' * delta_layers{layer+1}) .* (1-a{layer}.*a{layer});
            end

            % compute gradient of loss w.r.t. parameters
            for layer=length(nodeLayers)-1:-1:1 
                delta_weights{layer} = delta_layers{layer+1} * a{layer}';
                temp_b = sum(delta_layers{layer+1},2);
                delta_biases{layer} = repmat(temp_b,1,batchSize);
            end

            % update weights and biases
            for layer=length(nodeLayers)-1:-1:1 
               weights{layer} = weights{layer} - (eta/size(mini_train,2)) .* delta_weights{layer};
               biases{layer} = biases{layer} - (eta/size(mini_train,2)) .* delta_biases{layer};
            end
        end

        % get prediction MSE/Accuracy for all training data in this epoch using updated parameters
        a{1} = train; 

        % set biases based on number of columns in training set
        epoch_biases = {};
        for elem=1:length(biases)
            epoch_biases{elem} = repmat(biases{elem}(:,1),1,size(train,2));
        end

        % feed-forward
        for conn=1:length(nodeLayers)-1 % for each connections between layers
            z{conn} = weights{conn}*a{conn} + epoch_biases{conn}; % weighted sum
            a{conn+1} = logsig(z{conn}); % sigmoid transformation of z
        end
        epoch_mse = immse(a{end},label);

        % percentage correct and accuracy for this mini-batch
        [~,pred_indices] = max(a{end}); % indice for predicted class for each example
        [~,true_indices] = max(label); % indice for true class for each example
        perf = pred_indices == true_indices; % performance, 1 if correct, 0 if not
        epoch_num_correct = sum(perf); % number of correct predictions
        epoch_accu = epoch_num_correct/size(train,2);
        % report learning status for each epoch
        if mod(epoch,1) == 0
            fprintf('Epoch: %d, MSE: %d, Correct: %d/%d, Acc: %d\n',epoch,epoch_mse,epoch_num_correct,size(train,2),epoch_accu);
        end

        % terminate when accuracy is higher than 99%
        if epoch_accu > 0.99
           break; 
        end 
    end
end

