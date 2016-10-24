
function [weights, biases] = Expanded_NN(input,target,split,nodeLayers,numEpochs,batchSize,eta,trans,cost,lambda,mu)

    % divide dataset into train,validation and test sets
    [index_train,index_valid,index_test] = dividerand(length(input),split(1),split(2),split(3));
    train = input(:,index_train);
    validation = input(:,index_valid);
    test = input(:,index_test);

    train_label = target(:,index_train);
    validation_label = target(:,index_valid);
    test_label = target(:,index_test);

    % initiate lists for plotting
    epoch_list = []; 
    train_cost_list = [];
    valid_cost_list = [];
    test_cost_list = [];

    % placeholder for weights between different layers
    % the first elem is w between first and second layers
    % there will be length(nodeLayers)-1 columns
    weights = cell(1,length(nodeLayers)-1);

    % placeholder for biases between different layers
    biases = cell(1,length(nodeLayers)-1);

    % placeholder for velocities of all weights and biases
    velocity_w = cell(1,length(nodeLayers)-1);
    velocity_b = cell(1,length(nodeLayers)-1);

    % placeholder for weighted sum
    % the first elem is weighted sum of input nodes and first layer weights
    z = cell(1,length(nodeLayers)-1);
    %z_perf = cell(1,length(nodeLayers)-1); % for performance use

    % placeholder for activation nodes
    % the first elem is the input layer
    a = cell(1,length(nodeLayers));
    %a_perf = cell(1,length(nodeLayers)); % for performance use

    % placeholder for gradient of loss w.r.t. activation function for all
    % layers except first layer
    % so if we have 4-layered network, it then contains 3 elements
    delta_layers = cell(1,length(nodeLayers)-1);

    % placeholder for gradient of loss w.r.t. weights/biases, for all connects between
    % layers
    delta_weights = cell(1,length(nodeLayers)-1);
    delta_biases = cell(1,length(nodeLayers)-1);

    % initialize weights and biases between layers, and momentum velocity for
    % each w and b
    for i=1:length(nodeLayers)-1 % for each layer starting from the second layer    
        if isequal(trans, 'sigmoid') | isequal(trans, 'tanh') | isequal(trans, 'softmax')
            weights{i} = randn(nodeLayers(i+1),nodeLayers(i))/sqrt(nodeLayers(i+1)); 
            biases{i} = randn(nodeLayers(i+1),batchSize);
        elseif isequal(trans, 'relu')
            weights{i} = randn(nodeLayers(i+1),nodeLayers(i))/sqrt(nodeLayers(i+1)/2); 
            biases{i} = randn(nodeLayers(i+1),batchSize);
        end

        velocity_w{i} = zeros(size(weights{i}));
        velocity_b{i} = zeros(size(biases{i}));
    end

    fprintf('     |          TRAIN          ||          VALIDATION          ||          TEST          \n');
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('Ep   | Cost |     Corr     | Acc || Cost |     Corr     | Acc || Cost |     Corr     | Acc \n');
    fprintf('-----------------------------------------------------------------------------------------\n');

    % loop through epochs
    for epoch=1:numEpochs
        % shuffle data
        cat_data = cat(1,train,train_label); 
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
                if isequal(trans, 'sigmoid')
                     a{conn+1} = logsig(z{conn}); % sigmoid transformation of z 
                elseif isequal(trans, 'tanh')
                     if conn == length(nodeLayers)-1
                         a{conn+1} = softmax(z{conn}); % use softmax on the output layer
                     else
                         a{conn+1} = tanh(z{conn}); % use tanh for layers before output
                     end
                elseif isequal(trans, 'relu')
                     if conn == length(nodeLayers)-1
                         a{conn+1} = softmax(z{conn}); % use softmax on the output layer
                     else
                         a{conn+1} = z{conn};
                         a{conn+1}(a{conn+1}<=0) = 0; % use relu for layers before output layer
                     end
                elseif isequal(trans, 'softmax')
                     if conn == length(nodeLayers)-1
                         a{conn+1} = softmax(z{conn}); % use softmax on the output layer
                     else
                         a{conn+1} = logsig(z{conn}); % use sigmoid for layers before output layer
                     end
                end
            end

            % back-propagation  
            % gradient of loss (delta) w.r.t. activation function in layers
            % except the first layer.
            % If we have 3-layered network, then we compute delta for last two
            % layers
            if isequal(cost, 'quadratic')
                delta_layers{length(nodeLayers)} = (a{end}-mini_label) .* delta_activation(trans, a{end}, 'is_output_layer'); % delta for output nodes
                for layer=length(nodeLayers)-1:-1:2 % going backwards
                    delta_layers{layer} = (weights{layer}' * delta_layers{layer+1}) .* delta_activation(trans, a{layer}, 'not_output_layer');
                end
            elseif isequal(cost, 'cross-entropy')
                delta_layers{length(nodeLayers)} = a{end}-mini_label; % delta for output layer
                for layer=length(nodeLayers)-1:-1:2 % going backwards
                    delta_layers{layer} = (weights{layer}' * delta_layers{layer+1}) .* delta_activation(trans, a{layer}, 'not_output_layer');
                end
            elseif isequal(cost, 'log')
                delta_layers{length(nodeLayers)} = a{end}-mini_label; % delta for output layer
                for layer=length(nodeLayers)-1:-1:2 % going backwards
                    delta_layers{layer} = (weights{layer}' * delta_layers{layer+1}) .* delta_activation(trans, a{layer}, 'not_output_layer');
                end
            end

            % compute gradient of loss w.r.t. parameters, including L2
            % regularization
            for layer=length(nodeLayers)-1:-1:1 
                delta_weights{layer} = delta_layers{layer+1} * a{layer}' + lambda * weights{layer};
                temp_b = sum(delta_layers{layer+1},2);
                delta_biases{layer} = repmat(temp_b,1,batchSize);
            end

            % update weights and biases using momentum
            for layer=length(nodeLayers)-1:-1:1 

               velocity_w{layer} = mu * velocity_w{layer} - ((eta/size(mini_train,2)) .* delta_weights{layer});
               weights{layer} = weights{layer} + velocity_w{layer};

               velocity_b{layer} = mu * velocity_b{layer} - ((eta/size(mini_train,2)) .* delta_biases{layer});
               biases{layer} = biases{layer} + velocity_b{layer};
            end
        end

        [train_cost,train_num_correct,train_accu] = perfmance(train,train_label,nodeLayers,weights,biases,trans,cost,lambda);
        [valid_cost,valid_num_correct,valid_accu] = perfmance(validation,validation_label,nodeLayers,weights,biases,trans,cost,lambda);
        [test_cost,test_num_correct,test_accu] = perfmance(test,test_label,nodeLayers,weights,biases,trans,cost,lambda);
        % report learning status for each epoch
        epoch_list = [epoch_list, epoch]; % append epoch
        train_cost_list = [train_cost_list,train_cost];
        valid_cost_list = [valid_cost_list,valid_cost];
        test_cost_list = [test_cost_list,test_cost];

        if mod(epoch,1) == 0
            fprintf('%d  | %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f || %.3f | %d/%d | %.3f \n', ...
                epoch,train_cost,train_num_correct,size(train,2),train_accu, ...
                valid_cost,valid_num_correct,size(validation,2),valid_accu, ...
                test_cost,test_num_correct,size(test,2),test_accu);
        end

        % early stopping
        if valid_accu > 0.99
           break; 
        end
    end

    figure,
    plot(epoch_list,train_cost_list,'DisplayName','Train')
    hold on
    plot(epoch_list,valid_cost_list,'DisplayName','Validation')
    hold on
    plot(epoch_list,test_cost_list,'DisplayName','Test')
    legend('show');
end

