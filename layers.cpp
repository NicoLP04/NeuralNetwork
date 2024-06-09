#include "layers.h"

Dense::Dense(int inputSize, int outputSize, ActivationFunction activationFunction)
    : mInputSize(inputSize), mOutputSize(outputSize) {
    mWeights.resize(inputSize * outputSize);
    mBiases.resize(outputSize);
    mActivationFunction = activationFunction;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);
    // std::uniform_real_distribution<double> d(-2, 2);

    for (double& weight : mWeights) weight = d(gen);
    for (double& bias : mBiases) bias = d(gen);
}

std::vector<double> Dense::forward(std::vector<double> input) {
    if (input.size() != mInputSize) {
        throw std::invalid_argument("Input size must match layer input size.");
    }

    mInput = input;
    std::vector<double> output(mOutputSize);

    for (int i = 0; i < mOutputSize; i++) {
        output[i] = mBiases[i];
        for (int j = 0; j < mInputSize; j++) {
            output[i] += mWeights[i * mInputSize + j] * input[j];
        }
        output[i] = activationFunctions[mActivationFunction](output[i]);
    }

    if (mActivationFunction == SOFTMAX) {
        double sum = 0;
        for (double& value : output) sum += value;
        for (double& value : output) value /= sum;
    }

    return output;
}

std::vector<double> Dense::backward(std::vector<double> outputGradient, double learningRate) {
    if (outputGradient.size() != mOutputSize) {
        throw std::invalid_argument("Output gradient size must match layer output size.");
    }

    std::vector<double> delta(mOutputSize);
    for (int i = 0; i < mOutputSize; ++i) {
        delta[i] = outputGradient[i] * activationFunctionPrimes[mActivationFunction](mInput[i]);
    }

    std::vector<double> inputGradient(mInputSize, 0.0);
    for (int i = 0; i < mOutputSize; ++i) {
        for (int j = 0; j < mInputSize; ++j) {
            inputGradient[j] += mWeights[j * mOutputSize + i] * delta[i];
            mWeights[j * mOutputSize + i] -= learningRate * delta[i] * mInput[j];
        }
        mBiases[i] -= learningRate * delta[i];
    }

    return inputGradient;
}
