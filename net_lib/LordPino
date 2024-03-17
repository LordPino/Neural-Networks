class SimpleCNN:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Initialize weights and biases for the convolutional layer
        self.conv_filter_size = 3
        self.conv_stride = 1
        self.conv_pad = 1
        self.conv_output_size = ((input_shape[0] - self.conv_filter_size + 2 * self.conv_pad) // self.conv_stride) + 1
        self.conv_filters = np.random.randn(self.conv_filter_size, self.conv_filter_size, input_shape[2], 1) * 0.01
        self.conv_bias = np.zeros((1, 1, 1, 1))
        
        # Initialize weights and biases for the fully connected layers
        self.fc1_input_size = self.conv_output_size * self.conv_output_size
        self.fc1_output_size = 64
        self.fc1_weights = np.random.randn(self.fc1_input_size, self.fc1_output_size) * 0.01
        self.fc1_bias = np.zeros((1, self.fc1_output_size))
        
        self.fc2_input_size = self.fc1_output_size
        self.fc2_output_size = self.num_classes
        self.fc2_weights = np.random.randn(self.fc2_input_size, self.fc2_output_size) * 0.01
        self.fc2_bias = np.zeros((1, self.fc2_output_size))
    
    def convolution(self, input_data):
        # Perform convolution
        conv_output = np.zeros((input_data.shape[0], self.conv_output_size, self.conv_output_size, self.conv_filters.shape[-1]))
        for i in range(self.conv_output_size):
            for j in range(self.conv_output_size):
                input_slice = input_data[:, i*self.conv_stride:i*self.conv_stride+self.conv_filter_size, 
                                         j*self.conv_stride:j*self.conv_stride+self.conv_filter_size, :]
                conv_output[:, i, j, :] = np.sum(input_slice * self.conv_filters, axis=(1,2,3)) + self.conv_bias
        return conv_output
    
    def relu(self, input_data):
        return np.maximum(input_data, 0)
    
    def flatten(self, input_data):
        return input_data.reshape(input_data.shape[0], -1)
    
    def softmax(self, input_data):
        exp_scores = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        # Forward pass through the network
        conv_output = self.convolution(X)
        relu_output = self.relu(conv_output)
        flattened_output = self.flatten(relu_output)
        fc1_output = np.dot(flattened_output, self.fc1_weights) + self.fc1_bias
        fc1_relu_output = self.relu(fc1_output)
        fc2_output = np.dot(fc1_relu_output, self.fc2_weights) + self.fc2_bias
        softmax_output = self.softmax(fc2_output)
        return softmax_output
    
    def train(self, X, y, learning_rate=0.01, epochs=10, batch_size=32):
        # Train the model using mini-batch gradient descent
        num_samples = X.shape[0]
        for epoch in range(epochs):
            for i in range(0, num_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Forward pass
                output = self.forward_pass(X_batch)
                
                # Compute loss (cross-entropy)
                loss = -np.sum(np.log(output[np.arange(len(y_batch)), y_batch])) / batch_size
                
                # Backpropagation
                grad_loss = output
                grad_loss[np.arange(len(y_batch)), y_batch] -= 1
                grad_loss /= batch_size
                
                grad_fc2_weights = np.dot(fc1_relu_output.T, grad_loss)
                grad_fc2_bias = np.sum(grad_loss, axis=0, keepdims=True)
                grad_fc1_relu_output = np.dot(grad_loss, self.fc2_weights.T)
                grad_fc1_output = grad_fc1_relu_output * (fc1_output > 0)
                grad_fc1_weights = np.dot(flattened_output.T, grad_fc1_output)
                grad_fc1_bias = np.sum(grad_fc1_output, axis=0, keepdims=True)
                grad_flattened_output = np.dot(grad_fc1_output, self.fc1_weights.T)
                grad_relu_output = grad_flattened_output.reshape(relu_output.shape)
                grad_conv_output = grad_relu_output * (conv_output > 0)
                grad_conv_filters = np.zeros_like(self.conv_filters)
                grad_conv_bias = np.sum(grad_conv_output, axis=(0,1,2), keepdims=True)
                for i in range(self.conv_output_size):
                    for j in range(self.conv_output_size):
                        input_slice = X_batch[:, i*self.conv_stride:i*self.conv_stride+self.conv_filter_size, 
                                               j*self.conv_stride:j*self.conv_stride+self.conv_filter_size, :]
                        for k in range(self.conv_filters.shape[-1]):
                            grad_conv_filters[:, :, :, k] += np.sum(input_slice * grad_conv_output[:, i, j, k][:, None, None, None], axis=0)
                
                # Update weights and biases
                self.fc1_weights -= learning_rate * grad_fc1_weights
                self.fc1_bias -= learning_rate * grad_fc1_bias
                self.fc2_weights -= learning_rate * grad_fc2_weights
                self.fc2_bias -= learning_rate * grad_fc2_bias
                self.conv_filters -= learning_rate * grad_conv_filters
                self.conv_bias -= learning_rate * grad_conv_bias
                
                print(f'Epoch {epoch+1}/{epochs}, Batch {i//batch_size+1}/{num_samples//batch_size}, Loss: {loss:.4f}')

# Assuming you have your training data X_train (shape: (num_samples, height, width, channels)) and labels y_train
cnn_model = SimpleCNN(input_shape=(28, 28, 1), num_classes=10)
cnn_model.train(X_train, y_train, learning_rate=0.01, epochs=10, batch_size=32)





class FCNNSimulatingConv:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Parameters for simulating convolution
        self.conv_filter_size = 3
        self.conv_stride = 1
        self.conv_pad = 1
        self.conv_output_size = ((input_shape[0] - self.conv_filter_size + 2 * self.conv_pad) // self.conv_stride) + 1
        
        # Initialize weights and biases for the fully connected layers
        self.fc1_input_size = self.conv_output_size * self.conv_output_size
        self.fc1_output_size = 64
        self.fc1_weights = np.random.randn(self.fc1_input_size * self.conv_filter_size * self.conv_filter_size, self.fc1_output_size) * 0.01
        self.fc1_bias = np.zeros((1, self.fc1_output_size))
        
        self.fc2_input_size = self.fc1_output_size
        self.fc2_output_size = self.num_classes
        self.fc2_weights = np.random.randn(self.fc2_input_size, self.fc2_output_size) * 0.01
        self.fc2_bias = np.zeros((1, self.fc2_output_size))
    
    def simulate_convolution(self, input_data):
        # Perform convolution simulation
        conv_output = np.zeros((input_data.shape[0], self.conv_output_size, self.conv_output_size, self.conv_filter_size * self.conv_filter_size * input_data.shape[-1]))
        idx = 0
        for i in range(self.conv_output_size):
            for j in range(self.conv_output_size):
                input_slice = input_data[:, i*self.conv_stride:i*self.conv_stride+self.conv_filter_size, 
                                         j*self.conv_stride:j*self.conv_stride+self.conv_filter_size, :]
                conv_output[:, i, j, :] = input_slice.reshape(input_slice.shape[0], -1)
        return conv_output
    
    def relu(self, input_data):
        return np.maximum(input_data, 0)
    
    def flatten(self, input_data):
        return input_data.reshape(input_data.shape[0], -1)
    
    def softmax(self, input_data):
        exp_scores = np.exp(input_data - np.max(input_data, axis=1, keepdims=True))
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def forward_pass(self, X):
        # Forward pass through the network
        conv_output = self.simulate_convolution(X)
        relu_output = self.relu(conv_output)
        flattened_output = self.flatten(relu_output)
        fc1_output = np.dot(flattened_output, self.fc1_weights) + self.fc1_bias
        fc1_relu_output = self.relu(fc1_output)
        fc2_output = np.dot(fc1_relu_output, self.fc2_weights) + self.fc2_bias
        softmax_output = self.softmax(fc2_output)
        return softmax_output
    
    def train(self, X, y, learning_rate=0.01, epochs=10, batch_size=32):
        # Training implementation is same as before
        pass
fcnn_model = FCNNSimulatingConv(input_shape=(28, 28, 1), num_classes=10)
fcnn_model.train(X_train, y_train, learning_rate=0.01, epochs=10, batch_size=32)



