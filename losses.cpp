#include "losses.h"

#include <iostream>

double mse(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    double sum = 0;
    for (int i = 0; i < yTrue.size(); i++) {
        sum += pow(yTrue[i] - yPred[i], 2);
    }

    return sum / yTrue.size();
}

std::vector<double> msePrime(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    std::vector<double> gradient(yTrue.size());
    for (int i = 0; i < yTrue.size(); i++) {
        gradient[i] = 2 * (yPred[i] - yTrue[i]) / yTrue.size();
    }

    return gradient;
}

double binaryCrossEntropy(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    double sum = 0;
    for (int i = 0; i < yTrue.size(); i++) {
        std::cout << "log(yPred[i]): " << log(yPred[i]) << " log(1 - yPred[i]): " << log(1 - yPred[i]) << "\n";
        sum += yTrue[i] * log(yPred[i]) + (1 - yTrue[i]) * log(1 - yPred[i]);
    }

    return -sum / yTrue.size();
}

std::vector<double> binaryCrossEntropyPrime(std::vector<double> yTrue, std::vector<double> yPred) {
    if (yTrue.size() != yPred.size()) {
        throw std::invalid_argument("Vectors yTrue and yPred must be of the same size");
    }

    std::vector<double> gradient(yTrue.size());
    for (int i = 0; i < yTrue.size(); i++) {
        gradient[i] = ((1 - yTrue[i]) / (1 - yPred[i]) - yTrue[i] / yPred[i]) / yTrue.size();
    }

    return gradient;
}
