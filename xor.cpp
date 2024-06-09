#include <time.h>

#include <iostream>

#include "activations.h"
#include "neuralNetwork.h"

void train(NeuralNetwork &nn, std::vector<std::vector<double>> inputs, std::vector<std::vector<double>> outputs,
           int epochs, double learningRate, bool verbose = false) {
    clock_t start = clock();
    for (int epoch = 0; epoch <= epochs; epoch++) {
        int correct = 0;
        double error = 0;
        for (int i = 0; i < inputs.size(); i++) {
            std::vector<double> prediction = nn.forward(inputs[i]);
            double instanceError = lossFunctions[nn.mLossFunction](outputs[i], prediction);
            error += instanceError;
            std::vector<double> gradient = lossFunctionPrimes[nn.mLossFunction](outputs[i], prediction);
            nn.backward(gradient, learningRate);
            if (std::abs(outputs[i][0] - prediction[0]) < 0.1) {
                correct++;
            }
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
                      << " - Accuracy: " << (int)(correct * 100.0) / inputs.size() << "% \r";
            std::cout.flush();
        }
    }
    if (verbose) {
        std::cout << std::endl;
    }
}

int main() {
    NeuralNetwork nn(MSE);
    nn.addLayer(new Dense(2, 3, TANH)).addLayer(new Dense(3, 1, TANH));

    std::vector<std::vector<double>> inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    std::vector<std::vector<double>> outputs = {{0}, {1}, {1}, {0}};

    train(nn, inputs, outputs, 100000, 0.1, true);

    for (int i = 0; i < inputs.size(); i++) {
        std::vector<double> prediction = nn.forward(inputs[i]);
        std::cout << "Prediction: " << prediction[0] << " Ground truth: " << outputs[i][0] << std::endl;
    }

    return 0;
}
