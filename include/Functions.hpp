// functions.hpp

#pragma once

enum class InitType {
    Zero,
    Rand,
    He
};

enum class Activation {
    RELU,
    SIGMOID,
    LINEAR,
    SOFTMAX
};

enum class Loss {
    MSE,
    CROSS_ENTROPY
};