#include <time.h>

#include <iostream>

#include "activations.h"
#include "layers.h"
#include "neuralNetwork.h"

void train(NeuralNetwork &nn, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs,
           int epochs, double learningRate, bool verbose = false) {
    std::vector<int> indices(inputs.size());
    for (int i = 0; i < inputs.size(); i++) indices[i] = i;
    clock_t start = clock();
    for (int epoch = 0; epoch <= epochs; epoch++) {
        shuffle(indices.begin(), indices.end(), std::default_random_engine(0));
        int correct = 0;
        double error = 0;
        for (int i = 0; i < inputs.size(); i++) {
            int idx = indices[i];
            std::vector<double> prediction = nn.forward(inputs[idx]);
            double instanceError = lossFunctions[nn.mLossFunction](outputs[idx], prediction);
            error += instanceError;
            std::vector<double> gradient = lossFunctionPrimes[nn.mLossFunction](outputs[idx], prediction);
            nn.backward(gradient, learningRate);
        }
        if (verbose) {
            int barWidth = 70;

            std::cout << "[";
            int pos = barWidth * epoch / epochs;
            for (int i = 0; i < barWidth; ++i) {
                if (i < pos)
                    std::cout << "=";
                else if (i == pos)
                    std::cout << ">";
                else
                    std::cout << " ";
            }
            double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
            std::cout << "] " << int(epoch * 100.0 / epochs) << " %" << " - Time: " << elapsed << "s - Error: " << error
                      << "% \r";
            std::cout.flush();
        }
    }
    if (verbose) {
        std::cout << std::endl;
    }
}

int main() {
    NeuralNetwork nn(BINARY_CROSS_ENTROPY);
    nn.addLayer(new Dense(2, 3, SIGMOID)).addLayer(new Softmax(3, 2));

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> outputs = {{1, 0}, {0, 1}, {0, 1}, {1, 0}};

    train(nn, inputs, outputs, 100000, 0.001, true);

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> prediction = nn.forward(inputs[i]);
        std::cout << "Prediction: " << prediction[0] << " Ground truth: " << outputs[i][0] << std::endl;
    }

    return 0;
}
