#pragma once

#include <random>
#include <stdexcept>
#include <vector>

#include "activations.h"

class Layer {
   public:
    virtual std::vector<double> forward(std::vector<double> input) = 0;
    virtual std::vector<double> backward(std::vector<double> outputGradient, double learningRate) = 0;
};

class Dense : public Layer {
   public:
    Dense(int inputSize, int outputSize, ActivationFunction activationFunction);
    std::vector<double> forward(std::vector<double> input) override;
    std::vector<double> backward(std::vector<double> outputGradient, double learningRate) override;

   private:
    ActivationFunction mActivationFunction;
    std::vector<double> mWeights;
    std::vector<double> mBiases;
    std::vector<double> mInput;
    int mInputSize;
    int mOutputSize;
};
