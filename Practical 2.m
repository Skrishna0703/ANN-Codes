% McCulloch-Pitts Neural Network for ANDNOT Function

% Define input vectors (x1, x2)
inputs = [0 0; 0 1; 1 0; 1 1]; % [x1, x2]

% Define weights and threshold
w = [1 -1]; % Weights for x1 and NOT x2
theta = 1;  % Threshold

% Compute net input and output
net_input = inputs * w';  
outputs = net_input >= theta; % Apply step activation function

% Display results
disp('Weights of Neuron:');
disp(['w1 = ' num2str(w(1))]);
disp(['w2 = ' num2str(w(2))]);
disp(['Threshold: Theta = ' num2str(theta)]);
disp(' ');

disp('Output:');
disp(['w1 = ' num2str(w(1))]);
disp(['w2 = ' num2str(w(2))]);
disp(['Threshold: Theta = ' num2str(theta)]);
disp(' ');
disp('With Output of Neuron:');
disp(num2str(outputs'));
