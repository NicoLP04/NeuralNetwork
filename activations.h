#pragma once

#include <cmath>
#include <map>

double linear(double x);
double linearPrime(double x);

double sigmoid(double x);
double sigmoidPrime(double x);

double relu(double x);
double reluPrime(double x);

double tanhh(double x);
double tanhPrime(double x);

double softmax(double x);
double softmaxPrime(double x);

// enum of activation functions
enum ActivationFunction { LINEAR, SIGMOID, RELU, TANH, SOFTMAX };

// map
static std::map<ActivationFunction, double (*)(double)> activationFunctions = {
    {SIGMOID, sigmoid}, {LINEAR, linear}, {RELU, relu}, {TANH, tanhh}, {SOFTMAX, softmax},
};

static std::map<ActivationFunction, double (*)(double)> activationFunctionPrimes = {
    {SIGMOID, sigmoidPrime}, {LINEAR, linearPrime}, {RELU, reluPrime}, {TANH, tanhPrime}, {SOFTMAX, softmaxPrime},
};
