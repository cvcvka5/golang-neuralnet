# Go Neural Network (From Scratch)

A modular, lightweight neural network implementation built entirely in Go. This project explores the fundamentals of machine learning—forward propagation, backpropagation, and gradient descent—without using external ML frameworks.

## Key Features

* **Modular Activation & Loss**: Easily swap between Sigmoid, ReLU, or Identity functions using a registry pattern.
* **Fully Connected Layers**: Supports custom architectures with any number of hidden layers and neurons.
* **Backpropagation**: Manual implementation of the Chain Rule for weight and bias optimization.
* **Persistence**: Save and load trained "brains" (weights and biases) using JSON.
* **Predict & Evaluate API**: Clean separation between internal math logic and user-facing prediction methods.

## Project Structure

* `internal/`: Core engine logic (Network, Neuron, and Connection logic).
* `pkg/activation/`: Non-linear functions and their derivatives.
* `pkg/loss/`: Error calculation logic (e.g., Mean Squared Error).
* `dataset/`: Helpers for loading and shuffling training data.

## 🛠️ Usage

### Training the Network
```go
// Create a network: 3 inputs, 1 output, 0.1 learning rate, and two 8-neuron hidden layers
nn, _ := internal.New(3, 1, 0.1, 8, 8)

// Train loop
for _, data := range trainingData {
    loss, _ := nn.Evaluate(data.Input, data.Output)
    nn.Backward(data.Output)
}
```

### Making Predictions
```go
result, _ := nn.Predict([]float64{0.1, 0.7, 0.3})
fmt.Printf("Prediction: %.2f\n", result[0])
```

## Example: Color Classifier
The current implementation includes a dataset designed to train the network to distinguish between Warm and Cool colors based on RGB values.

---

#### Developer Note
This project was built to understand the "magic" behind the math. Every gradient update and weight tweak is handled manually to ensure a deep understanding of how information flows through a neural network.