function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

%% 代价函数

% y转换为矩阵
y_vec = zeros(num_labels, m);
for i = 1:m
    y_vec(y(i),i) = 1;
end

% 计算假设函数h_theta
a_1 = [ones(1,m);X']; % 添加偏置1，维度401*5000

z_2 = Theta1 * a_1; % 维度25*5000
a_2 = [ones(1,m);sigmoid(z_2)]; % 添加偏置1，维度26*5000

z_3 = Theta2 * a_2; % 维度10*5000
a_3 = sigmoid(z_3); % 添加偏置1，维度10*5000
h_theta = a_3;

% 计算代价函数
J = sum(sum(-y_vec .* log(h_theta) - (1-y_vec) .* log(1 - h_theta)))/m;

% 正则项
J = J + lambda * sum(sum(Theta1(:,2:end).^2)) / (2*m);
J = J + lambda * sum(sum(Theta2(:,2:end).^2)) / (2*m);


%% 反向传播

epsilon_3 = a_3 - y_vec; % 维度10*5000
epsilon_2 = Theta2(:,2:end)' * epsilon_3 .* sigmoidGradient(z_2); % 维度 25*5000

delta_2 = epsilon_3*a_2' ./ m;  % 维度10*26
delta_1 = epsilon_2*a_1' ./ m; % 维度25*401

% 正则项
Theta1(:,1) = 0;
Theta2(:,1) = 0;
regularization_1 = lambda*Theta1/m;
regularization_2 = lambda*Theta2/m;

Theta1_grad = delta_1 + regularization_1;
Theta2_grad = delta_2 + regularization_2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
