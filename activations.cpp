#include "activations.h"

double linear(double x) { return x; }
double linearPrime(double x) { return 1; }

double sigmoid(double x) { return 1 / (1 + exp(-x)); }
double sigmoidPrime(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

double relu(double x) { return x > 0 ? x : 0; }
double reluPrime(double x) { return x > 0 ? 1 : 0; }

double tanhh(double x) { return std::tanh(x); }
double tanhPrime(double x) { return 1 - std::pow(tanh(x), 2); }

double softmax(double x) { return exp(x); }
double softmaxPrime(double x) { return exp(x) / pow(exp(x), 2); }
