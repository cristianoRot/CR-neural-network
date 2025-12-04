# CrNeuralNet

A lightweight, high-performance neural network library implemented in C++17. Designed for educational purposes and practical machine learning tasks.

## Features

- **Pure C++17**: No external dependencies except standard library and Accelerate framework
- **Multiple Activation Functions**: ReLU, Softmax, Linear, Sigmoid (planned)
- **Loss Functions**: Cross-Entropy, Mean Squared Error (MSE)
- **Weight Initialization**: Zero, Random, He initialization
- **Optimization**: Momentum-based gradient descent
- **Learning Rate Scheduling**: Automatic reduction on plateau
- **Model Persistence**: Save and load trained models
- **Real-time Training Visualization**: Live loss and accuracy graphs
- **CSV Dataset Support**: Easy data loading from CSV files
- **Batch Training**: Efficient batch processing for improved performance
- **Accelerate Framework**: Optimized matrix operations using Apple's Accelerate

## Quick Start

### Building the Project

```bash
make clean
make
```

This will build three executables:
- `build/train` - Training script
- `build/test` - Test/Evaluation script
- `build/main` - Main application

### Training a Model

```bash
make train
```

### Testing a Model

```bash
make test
```

### Running the Main Application

```bash
make run
```

## Example: MNIST Digit Classification

### Results

Trained on MNIST dataset with the following results:
- **Training Accuracy**: 99.1983%
- **Test Accuracy**: 95.39%

### Dataset Preparation

The MNIST dataset should be in CSV format with:
- First column: `label` (0-9)
- Remaining columns: Pixel values (0-255), flattened 28x28 images
- Values are automatically normalized to [0, 1] by dividing by 255

Example CSV structure:
```
label,1x1,1x2,1x3,...,28x28
5,0,0,0,...,0
0,0,0,0,...,0
...
```

### Training

```cpp
#include "include/Network.hpp"
#include "include/Dataset.hpp"

int main() {
    // Load training data
    Dataset dataset = Dataset::from_csv(
        "data/mnist_train.csv",
        {"ALL"},  // Use all columns except 'label'
        "label"
    );
    
    // Create network: 784 inputs -> 64 -> 32 -> 10 outputs
    Network network(
        {
            Layer(784, 64, Activation::RELU),
            Layer(64, 32, Activation::RELU),
            Layer(32, 10, Activation::SOFTMAX)
        },
        0.001,           // Learning rate
        InitType::He,    // Weight initialization
        Loss::CROSS_ENTROPY
    );
    
    // Train for 100 epochs with batch size 32
    network.train(dataset, 100, 32);
    
    // Save the model
    network.save("models/mnist_model.crnn");
    
    return 0;
}
```

### Testing/Evaluation

The test script (`test.cpp`) evaluates the model on the test dataset:

```cpp
#include "include/Network.hpp"
#include "include/Dataset.hpp"

int main() {
    // Load test data
    Dataset test_dataset = Dataset::from_csv(
        "data/mnist_test.csv",
        {"ALL"},
        "label"
    );
    
    // Create network with same architecture
    Network network(
        {
            Layer(784, 64, Activation::RELU),
            Layer(64, 32, Activation::RELU),
            Layer(32, 10, Activation::SOFTMAX)
        },
        0.001,
        InitType::He,
        Loss::CROSS_ENTROPY
    );
    
    // Load trained weights
    network.load("models/mnist_model.crnn");
    
    // Evaluate
    size_t correct = 0;
    for (size_t i = 0; i < test_dataset.size(); i++) {
        Matrix input = test_dataset.get_input(i, 1);
        network.forward(input);
        const Matrix& output = network.get_output();
        size_t predicted = output.argmax_col(0);
        std::vector<size_t> labels = test_dataset.get_output(i, 1);
        if (predicted == labels[0]) correct++;
    }
    
    double accuracy = static_cast<double>(correct) / test_dataset.size();
    std::cout << "\033[34mAccuracy: " << accuracy * 100 << "%\033[0m" << std::endl;
    
    return 0;
}
```

Run with:
```bash
make test
```

## Training Features

### Real-time Visualization

During training, you'll see:
- **Loss Graph (Red)**: Shows training loss over epochs
- **Accuracy Graph (Blue)**: Shows training accuracy over epochs
- **Current Metrics**: Epoch number, current/max accuracy, and loss

The graphs update in-place, providing real-time feedback on training progress.

### Learning Rate Scheduling

The network automatically reduces the learning rate when accuracy plateaus:
- **Patience**: 20 epochs (default)
- **Factor**: 0.7 (multiplies LR by this factor)
- **Min LR**: 1e-6 (minimum learning rate)

### Model Checkpointing

The best model (highest accuracy) is automatically saved to:
- `checkpoints/model.crnn`

You can also manually save models:
```cpp
network.save("path/to/model.crnn");
```

## File Structure

```
CrNeuralNet/
├── include/
│   ├── Network.hpp
│   ├── Layer.hpp
│   ├── Matrix.hpp
│   ├── Dataset.hpp
│   ├── Functions.hpp
│   ├── TrainingLogger.hpp
│   └── ModelIO.hpp
├── src/
│   ├── Network.cpp
│   ├── Layer.cpp
│   ├── Matrix.cpp
│   ├── Dataset.cpp
│   ├── ModelIO.cpp
│   └── TrainingLogger.cpp
├── data/
│   ├── mnist_train.csv
│   └── mnist_test.csv
├── train.cpp      # Training script (run with: make train)
├── test.cpp       # Test/Evaluation script (run with: make test)
├── main.cpp       # Main application (run with: make run)
├── Makefile
└── README.md
```

## Available Make Targets

- `make` or `make all` - Build all executables (train, test, main)
- `make train` - Build and run the training script
- `make test` - Build and run the test/evaluation script
- `make run` - Build and run the main application
- `make clean` - Remove all build files
- `make rebuild` - Clean and rebuild everything
- `make help` - Show available targets

## Performance Optimizations

- **Accelerate Framework**: Matrix operations use Apple's Accelerate framework for optimized BLAS/vDSP operations
- **Batch Processing**: Supports batch training (default: 32) for improved performance
- **In-place Operations**: Minimizes memory allocations during training
- **Move Semantics**: Efficient matrix copying and moving

## License

This project is for educational purposes.
