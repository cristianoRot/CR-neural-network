# CrNeuralNet

A lightweight C++ implementation of a neural network library with matrix operations, layer functionality, and automatic differentiation for backpropagation.

## Features

- **Matrix Operations**: Addition, subtraction, multiplication, transpose, Hadamard product
- **Activation Functions**: ReLU, Softmax, and their derivatives
- **Neural Network Layers**: Input, Hidden, and Output layers with forward/backward propagation
- **Weight Initialization**: Xavier, He, Random, and Zero initialization methods
- **Training**: Complete training loop with gradient descent and accuracy tracking
- **CSV Dataset Loading**: Load training data directly from CSV files with automatic label mapping
- **Learning Rate Scheduling**: Automatic learning rate reduction on plateau for better convergence
- **Momentum Optimization**: Gradient descent with momentum for faster training
- **Modern C++**: Uses std::vector for memory management, operator overloading, and smart pointers

## Project Structure

```
CrNeuralNet/
├── include/
│   ├── Matrix.hpp      # Matrix class definition
│   ├── Layer.hpp       # Layer hierarchy (LayerBase, Layer, HiddenLayer, OutputLayer)
│   ├── Network.hpp     # Network class definition
│   ├── Dataset.hpp     # Dataset class for training data
│   └── InitType.hpp    # Weight initialization types
├── src/
│   ├── Matrix.cpp      # Matrix implementation
│   ├── Layer.cpp       # Layer implementation
│   ├── Network.cpp     # Network implementation
│   └── Dataset.cpp     # Dataset implementation
├── build/              # Object files directory (created during compilation)
├── data/               # Dataset files (CSV format)
│   └── iris.csv        # Iris flower dataset example
├── main.cpp            # Main example program
├── Makefile           # Build automation
└── README.md
```

## Compilation

### Using Makefile (Recommended)

The project includes a Makefile for easy compilation and management:

```bash
# Compile the entire project
make

# Clean build files
make clean

# Rebuild everything (clean + build)
make rebuild

# Build and run the test
make run

# Show available commands
make help
```

## Usage

### Loading Dataset from CSV

The library supports loading datasets directly from CSV files. This is particularly useful for real-world datasets:

```cpp
#include "include/Network.hpp"
#include "include/Dataset.hpp"

int main() {
    // Load dataset from CSV file
    // Parameters: file path, input column names, output column name
    Dataset dataset = Dataset::from_csv(
        "data/iris.csv",
        {"sepal_length", "sepal_width", "petal_length", "petal_width"},
        "species"
    );
    
    // The function automatically:
    // - Parses the CSV header to find column indices
    // - Converts numeric input columns to double values
    // - Maps text labels (like "setosa", "versicolor") to numeric indices (0, 1, 2)
    // - Handles whitespace and formatting automatically
}
```

### Iris Flower Classification Example

Complete example using the Iris dataset (150 samples, 4 features, 3 classes):

```cpp
#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <iostream>

int main() {
    // Load dataset: 4 features (sepal/petal length/width) -> 3 classes (setosa/versicolor/virginica)
    Dataset dataset = Dataset::from_csv(
        "data/iris.csv",
        {"sepal_length", "sepal_width", "petal_length", "petal_width"},
        "species"
    );
    
    // Network: 4 input -> 4 hidden -> 3 output
    Network network({4, 4, 3}, InitType::He, 0.01);
    
    std::cout << "Starting training..." << std::endl;
    network.train(dataset, 1000);
    std::cout << "\nTraining completed!" << std::endl;
    
    return 0;
}
```

The network automatically maps text labels ("setosa", "versicolor", "virginica") to numeric indices (0, 1, 2) and typically achieves 95-99% accuracy.

### Manual Dataset Creation

You can also create datasets manually:

```cpp
std::vector<Matrix> inputs;
std::vector<size_t> outputs;

inputs.push_back(Matrix(2, 1, {0.0, 0.0}));
outputs.push_back(0);
// ... add more examples

Dataset dataset(inputs, outputs);
Network network({2, 4, 2}, InitType::He, 0.1);
network.train(dataset, 100);
```

### Matrix Operations

```cpp
#include "include/Matrix.hpp"

// Create matrices
Matrix m1(2, 3, {1, 2, 3, 4, 5, 6});
Matrix m2(3, 2, {1, 2, 3, 4, 5, 6});

// Matrix operations
Matrix result = m1 * m2;           // Matrix multiplication
Matrix sum = m1 + m2.transpose();  // Addition with transpose
Matrix relu = m1.relu();           // ReLU activation
Matrix softmax = m1.softmax();    // Softmax activation

// Print matrix
result.print();
```

## Weight Initialization

The network supports different weight initialization methods:

- `InitType::Zero` - Initialize all weights to zero (not recommended for training)
- `InitType::Rand` - Random initialization with small values (-0.01 to 0.01)
- `InitType::He` - He initialization (recommended for ReLU activation functions)
- `InitType::Xavier` - Xavier/Glorot initialization (good for tanh/sigmoid)

## Training Features

### Learning Rate Reduction on Plateau

The network automatically reduces the learning rate when the accuracy stops improving:

- **Patience**: 20 epochs (waits 20 epochs without improvement)
- **Factor**: 0.7 (reduces LR to 70% of current value)
- **Min Learning Rate**: 1e-6 (minimum allowed learning rate)
- **Min Delta**: 0.001 (minimum improvement to reset patience counter)

This helps the network converge more reliably and achieve better final accuracy.

### Momentum Optimization

Gradient descent uses momentum (beta=0.9) to smooth out updates and accelerate convergence in the right direction.

## Requirements

- **Compiler**: C++17 compatible compiler (g++, clang++)
- **Platform**: macOS (uses Accelerate framework for optimized operations)
- **Dependencies**: None (uses only standard library)

## Performance

The implementation uses:
- **Accelerate Framework**: For optimized matrix operations on macOS
- **Efficient Memory Management**: std::vector with pre-allocated sizes
- **Smart Pointers**: std::unique_ptr for automatic memory management

## License

This project is open source and available under the MIT License.
