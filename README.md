# CrNeuralNet

A lightweight C++ implementation of a neural network library with layer functionality, automatic differentiation for backpropagation, and real-time training visualization.

## Features

- **Activation Functions**: ReLU, Softmax, and their derivatives
- **Neural Network Layers**: Flexible layer architecture with forward/backward propagation
- **Weight Initialization**: He, Random, and Zero initialization methods
- **Training**: Complete training loop with gradient descent and accuracy tracking
- **CSV Dataset Loading**: Load training data directly from CSV files with automatic label mapping
- **Learning Rate Scheduling**: Automatic learning rate reduction on plateau for better convergence
- **Momentum Optimization**: Gradient descent with momentum for faster training
- **Real-time Visualization**: Live loss and accuracy graphs with color-coded plots during training
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
│   ├── iris.csv        # Iris flower dataset example
│   └── btc_data.csv    # Bitcoin dataset example
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

### Bitcoin Dataset Example

Example using the Bitcoin dataset for binary classification:

```cpp
#include "include/Network.hpp"
#include "include/Dataset.hpp"
#include <iostream>

int main() {
    // Load Bitcoin dataset: 24 features -> 2 classes (binary classification)
    Dataset dataset = Dataset::from_csv(
        "data/btc_data.csv",
        {"ALL"},  // Use all columns except the output column
        "label"
    );
    
    // Network architecture: 24 input -> 12 hidden -> 6 hidden -> 2 output
    Network network(
        {
            Layer(24, 12, Activation::RELU),
            Layer(12, 6, Activation::RELU),
            Layer(6, 2, Activation::SOFTMAX)
        },
        InitType::He,
        0.01
    );
    
    network.train(dataset, 50);
    
    return 0;
}
```

The `{"ALL"}` parameter automatically selects all columns except the output column as input features. The training process displays real-time graphs showing loss (red) and accuracy (blue) trends.

### Other Datasets

You can also use other datasets like the Iris flower dataset (150 samples, 4 features, 3 classes) by loading them from CSV files. The library automatically handles text label mapping to numeric indices.

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

## Training Visualization

During training, the library displays real-time graphs showing:
- **Loss Graph (Red)**: Cross-entropy or MSE loss over epochs
- **Accuracy Graph (Blue)**: Classification accuracy over epochs

The graphs update in-place during training, showing the current epoch, max accuracy achieved, and both metrics side-by-side. The visualization uses a fixed scale based on initial values to clearly show improvement trends.

## Weight Initialization

The network supports different weight initialization methods:

- `InitType::Zero` - Initialize all weights to zero (not recommended for training)
- `InitType::Rand` - Random initialization with small values (-0.01 to 0.01)
- `InitType::He` - He initialization (recommended for ReLU activation functions)

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
