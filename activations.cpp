#include "activations.h"

#include <iostream>

// Linear activation function
std::vector<double> linear(const std::vector<double>& X) { return X; }
std::vector<double> linearPrime(const std::vector<double>& X) { return std::vector<double>(X.size(), 1); }

// Sigmoid activation function
std::vector<double> sigmoid(const std::vector<double>& X) {
    std::vector<double> res(X.size());
    for (int i = 0; i < X.size(); i++) {
        res[i] = 1 / (1 + exp(-X[i]));
    }
    return res;
}
std::vector<double> sigmoidPrime(const std::vector<double>& X) {
    std::vector<double> res(X.size());
    for (int i = 0; i < X.size(); i++) {
        double sig = 1 / (1 + exp(-X[i]));
        res[i] = sig * (1 - sig);
    }
    return res;
}

// ReLU activation function
std::vector<double> relu(const std::vector<double>& X) {
    std::vector<double> res(X.size());
    for (int i = 0; i < X.size(); i++) {
        res[i] = X[i] > 0 ? X[i] : 0;
    }
    return res;
}
std::vector<double> reluPrime(const std::vector<double>& X) {
    std::vector<double> res(X.size());
    for (int i = 0; i < X.size(); i++) {
        res[i] = X[i] > 0 ? 1 : 0;
    }
    return res;
}

// Tanh activation function
std::vector<double> tanhh(const std::vector<double>& X) {
    std::vector<double> res(X.size());
    for (int i = 0; i < X.size(); i++) {
        res[i] = tanh(X[i]);
    }
    return res;
}
std::vector<double> tanhPrime(const std::vector<double>& X) {
    std::vector<double> res(X.size());
    for (int i = 0; i < X.size(); i++) {
        res[i] = 1 - pow(tanh(X[i]), 2);
    }
    return res;
}

// Softmax activation function
std::vector<double> softmax(const std::vector<double>& X) {
    double max = *std::max_element(X.begin(), X.end());
    std::vector<double> res(X.size());
    double sum = 0;
    for (int i = 0; i < X.size(); i++) {
        res[i] = exp(X[i] - max);
        sum += res[i];
    }
    for (int i = 0; i < X.size(); i++) {
        res[i] /= sum;
    }
    return res;
}

// Softmax derivative (Jacobean matrix)
std::vector<double> softmaxPrime(const std::vector<double>& X) {
    std::vector<double> s = softmax(X);
    std::vector<std::vector<double>> jacobian(X.size(), std::vector<double>(X.size()));

    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X.size(); j++) {
            if (i == j) {
                jacobian[i][j] = s[i] * (1 - s[i]);
            } else {
                jacobian[i][j] = -s[i] * s[j];
            }
        }
    }

    std::vector<double> result(X.size() * X.size(), 0.0);
    for (int i = 0; i < X.size(); i++) {
        for (int j = 0; j < X.size(); j++) {
            result[i * X.size() + j] = jacobian[i][j];
        }
    }

    return result;
}
