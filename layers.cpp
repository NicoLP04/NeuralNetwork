#include "layers.h"

#include "activations.h"

Dense::Dense(int inputSize, int outputSize, ActivationFunction activationFunction)
    : mInputSize(inputSize), mOutputSize(outputSize) {
    mOutput.resize(outputSize);
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

std::vector<double> Dense::forward(const std::vector<double>& input) {
    if (input.size() != mInputSize) {
        throw std::invalid_argument("Input size must match layer input size.");
    }

    mInput = input;

    for (int i = 0; i < mOutputSize; i++) {
        mOutput[i] = mBiases[i];
        for (int j = 0; j < mInputSize; j++) {
            mOutput[i] += mWeights[i * mInputSize + j] * input[j];
        }
    }

    return activationFunctions[mActivationFunction](mOutput);
}

std::vector<double> Dense::backward(const std::vector<double>& outputGradient, double learningRate) {
    if (outputGradient.size() != mOutputSize) {
        throw std::invalid_argument("Output gradient size must match layer output size.");
    }

    // compute activation prime
    std::vector<double> activatedOutput = activationFunctionPrimes[mActivationFunction](mOutput);
    std::vector<double> inputGradient(mInputSize, 0.0);

    // Compute input gradient and weight gradients
    for (int i = 0; i < mOutputSize; ++i) {
        activatedOutput[i] *= outputGradient[i];
        for (int j = 0; j < mInputSize; ++j) {
            inputGradient[j] += mWeights[i * mInputSize + j] * activatedOutput[i];
            mWeights[i * mInputSize + j] -= learningRate * activatedOutput[i] * mInput[j];
        }
        mBiases[i] -= learningRate * activatedOutput[i];
    }

    return inputGradient;
}

Softmax::Softmax(int inputSize, int outputSize) : Dense(inputSize, outputSize, ActivationFunction::SOFTMAX) {}

std::vector<double> Softmax::forward(const std::vector<double>& input) { return Dense::forward(input); }

std::vector<double> Softmax::backward(const std::vector<double>& outputGradient, double learningRate) {
    // compute jacobian matrix
    std::vector<double> jacobian(mOutputSize * mOutputSize, 0.0);
    std::vector<double> s = activationFunctions[mActivationFunction](mOutput);

    for (int i = 0; i < mOutputSize; ++i) {
        for (int j = 0; j < mOutputSize; ++j) {
            if (i == j) {
                jacobian[i * mOutputSize + j] = s[i] * (1 - s[i]);
            } else {
                jacobian[i * mOutputSize + j] = -s[i] * s[j];
            }
        }
    }

    std::vector<double> activatedOutput(mOutputSize, 0.0);

    for (int i = 0; i < mOutputSize; ++i) {
        for (int j = 0; j < mOutputSize; ++j) {
            activatedOutput[i] += outputGradient[j] * jacobian[j * mOutputSize + i];
        }
    }

    // compute output gradient
    std::vector<double> inputGradient(mInputSize, 0.0);

    // Compute input gradient and weight gradients
    for (int i = 0; i < mOutputSize; ++i) {
        for (int j = 0; j < mInputSize; ++j) {
            inputGradient[j] += mWeights[i * mInputSize + j] * activatedOutput[i];
            mWeights[i * mInputSize + j] -= learningRate * activatedOutput[i] * mInput[j];
        }
        mBiases[i] -= learningRate * activatedOutput[i];
    }

    return inputGradient;
}
