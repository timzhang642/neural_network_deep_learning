% calculate partial derivative of the output of the transformation function, a
% trans is a string variable indicates which transformation function it is
% a is the activation output
% is_output_layer is a boolean variable used for softmax function only
function a_delta = delta_activation(trans, a, is_output_layer)
    if isequal(trans, 'sigmoid')
        a_delta = (1 - a) .* a;
    elseif isequal(trans, 'tanh')
        a_delta = 1 - a.^2;
    elseif isequal(trans, 'relu')
        a_delta = a;
        a_delta(a_delta>0) = 1;
    elseif isequal(trans, 'softmax')
        if isequal(is_output_layer, 'is_output_layer')
            a_delta = ones(size(a)); 
        else
            a_delta = (1 - a) .* a; % return the derivative of sigmoid if not the last layer
        end
    end
end