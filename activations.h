#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

std::vector<double> linear(const std::vector<double>& X);
std::vector<double> linearPrime(const std::vector<double>& X);

std::vector<double> sigmoid(const std::vector<double>& X);
std::vector<double> sigmoidPrime(const std::vector<double>& X);

std::vector<double> relu(const std::vector<double>& X);
std::vector<double> reluPrime(const std::vector<double>& X);

std::vector<double> tanhh(const std::vector<double>& X);
std::vector<double> tanhPrime(const std::vector<double>& X);

std::vector<double> softmax(const std::vector<double>& X);
std::vector<double> softmaxPrime(const std::vector<double>& X);

// enum of activation functions
enum ActivationFunction { LINEAR, SIGMOID, RELU, TANH, SOFTMAX };

// map
static std::map<ActivationFunction, std::vector<double> (*)(const std::vector<double>& X)> activationFunctions = {
    {SIGMOID, sigmoid}, {LINEAR, linear}, {RELU, relu}, {TANH, tanhh}, {SOFTMAX, softmax},
};

static std::map<ActivationFunction, std::vector<double> (*)(const std::vector<double>& X)> activationFunctionPrimes = {
    {SIGMOID, sigmoidPrime}, {LINEAR, linearPrime}, {RELU, reluPrime}, {TANH, tanhPrime}, {SOFTMAX, softmaxPrime},
};
