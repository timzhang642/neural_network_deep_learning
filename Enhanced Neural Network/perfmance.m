% return prediction performance
function [epoch_cost,epoch_num_correct,epoch_accu] = perfmance(train,label,nodeLayers,weights,biases,trans,cost,lambda)
    z_perf = cell(1,length(nodeLayers)-1); % for performance use
    a_perf = cell(1,length(nodeLayers)); % for performance use
    
    % get prediction performance for all training data in this epoch using updated parameters
    a_perf{1} = train; 
    
    % set biases matrix that corresponds to the size of the training set
    epoch_biases = {};
    for elem=1:length(biases)
        epoch_biases{elem} = repmat(biases{elem}(:,1),1,size(train,2));
    end
    
    % feed-forward
    for conn=1:length(nodeLayers)-1 % for each connections between layers
        z_perf{conn} = weights{conn}*a_perf{conn} + epoch_biases{conn}; % weighted sum
        if isequal(trans, 'sigmoid')
             a_perf{conn+1} = logsig(z_perf{conn}); % sigmoid transformation of z
        elseif isequal(trans, 'tanh')
             if conn == length(nodeLayers)-1
                 a_perf{conn+1} = softmax(z_perf{conn}); % use softmax on the output layer
             else
                 a_perf{conn+1} = tanh(z_perf{conn}); % use tanh for layers before output
             end
        elseif isequal(trans, 'relu')
             if conn == length(nodeLayers)-1
                 a_perf{conn+1} = softmax(z_perf{conn}); % use softmax on the output layer
             else
                 a_perf{conn+1} = z_perf{conn};
                 a_perf{conn+1}(a_perf{conn+1}<=0) = 0; % use relu for layers before output layer
             end
        elseif isequal(trans, 'softmax')
             if conn == length(nodeLayers)-1
                 a_perf{conn+1} = softmax(z_perf{conn}); % use softmax on the output layer
             else
                 a_perf{conn+1} = logsig(z_perf{conn}); % use sigmoid for layers before output layer
             end
        end
    end
    
    % calculate cost
    % L2 regularization cost
    parameters_squared = 0;
    for i=1:length(weights)
        parameters_squared = parameters_squared + sum(sum(weights{i}.^2)) + sum(biases{i}(:,1).^2); % sum of squared weights and biases
    end
    reg_loss = 0.5 * lambda * parameters_squared/size(label,2); % L2 cost
    % prediction cost
    if isequal(cost, 'quadratic')
        epoch_cost = immse(a_perf{end},label) + reg_loss; % MSE cost
    elseif isequal(cost, 'cross-entropy')
        epoch_cost = (-sum(sum(label .* log(a_perf{end}) + (1-label) .* log(1-a_perf{end}))))/size(label,2) + reg_loss; % cross-entropy for binary class
    elseif isequal(cost, 'log')
        epoch_cost = (sum(sum(label .* -log(a_perf{end}))))/size(label,2) + reg_loss; % cross-entropy for multiclass
    end

    % percentage correct and accuracy for this mini-batch
    [~,pred_indices] = max(a_perf{end}); % indice for predicted class for each example
    [~,true_indices] = max(label); % indice for true class for each example
    perf = pred_indices == true_indices; % performance, 1 if correct, 0 if not
    epoch_num_correct = sum(perf); % number of correct predictions
    epoch_accu = epoch_num_correct/size(train,2); 
end