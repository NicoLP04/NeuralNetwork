#pragma once

#include <random>
#include <stdexcept>
#include <vector>

#include "activations.h"

class Layer {
   public:
    virtual std::vector<double> forward(const std::vector<double>& input) = 0;
    virtual std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) = 0;
};

class Dense : public Layer {
   public:
    Dense(int inputSize, int outputSize, ActivationFunction activationFunction);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;

   protected:
    ActivationFunction mActivationFunction;
    std::vector<double> mWeights;
    std::vector<double> mBiases;
    std::vector<double> mInput;
    std::vector<double> mOutput;
    int mInputSize;
    int mOutputSize;
};

class Softmax : public Dense {
   public:
    Softmax(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double>& input) override;
    std::vector<double> backward(const std::vector<double>& outputGradient, double learningRate) override;
};
