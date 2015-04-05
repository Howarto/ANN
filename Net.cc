#include "Net.hh"

Net::Net(int inputSize, int hiddenSize, int outputSize) {
    createNetwork(inputSize, hiddenSize, outputSize);
}

void Net::createNetwork(int inputSize, int hiddenSize, int outputSize) {
    lastDeltas = vector<vector<vector<double> > >();

    if (hiddenSize > 0) {
        layers = vector<vector<vector<double> > >(2);
        createLayer(hiddenSize, inputSize, 0);
        createLayer(outputSize, hiddenSize, 1);
    }
    else {
        layers = vector<vector<vector<double> > >(1);
        createLayer(outputSize, inputSize, 0);
    }
}

void Net::createLayer(int size, int inputSize, int layerIndex) {
    layers[layerIndex] = vector<vector<double> >(size);
    int numberOfWeights = inputSize + 1;        // +1 because of BIAS

    for (int neuronIndex = 0; neuronIndex < size; ++neuronIndex) {
        layers[layerIndex][neuronIndex] = vector<double>(numberOfWeights);

        for (int weightIndex = 0; weightIndex < numberOfWeights; ++weightIndex) {
            layers[layerIndex][neuronIndex][weightIndex] = getRandomWeight(numberOfWeights);
        }
    }
}

double Net::getRandomWeight(int numberOfWeights) {
    // Returns a random in the interval [-1/sqrt(numberOfWeights), 1/sqrt(numberOfWeights)]
    double x = 1 / sqrt(numberOfWeights);
    double random = (double)rand() / RAND_MAX;
    return random * 2 * x - x;
}

void Net::compute(const vector<double> &input, vector<double> &results) {
    vector<double> parsedInputs(input);
    allInputs = vector<vector<double> >(layers.size());
    allResults = vector<vector<double> >(layers.size());

    for (int layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        parsedInputs.push_back(1);           // We add the BIAS value 1 to the previous input data.
        results = vector<double>(layers[layerIndex].size());

        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); ++neuronIndex) {
            results[neuronIndex] = computeNeuron(parsedInputs, layers[layerIndex][neuronIndex], false);
        }

        allInputs[layerIndex] = vector<double>(parsedInputs);
        allResults[layerIndex] = vector<double>(results);
        parsedInputs = vector<double>(results);
    }
}

double Net::computeNeuron(const vector<double> &inputs, const vector<double> &weights, bool derivative) {
    double output = 0;

    for (int i = 0; i < inputs.size(); ++i) {
        output += inputs[i] * weights[i];
    }

    return activationFunction(output, derivative);
}

double Net::activationFunction(double x, bool derivative) {
    if (derivative) {
        return activationFunction(x, false) * (1 - activationFunction(x, false));
    }

    return 1 / (1 + exp(-x));
}

void Net::train(const vector<double> &input, const vector<double> &expectedOutputs, double trainingSpeed, double momentum) {
    // At first, we compute the current results, to know what error we have.
    vector<double> notUsedAgainResults;
    compute(input, notUsedAgainResults);

    // Calculating errors (difference between expected results and current results).
    vector<vector<double> > networkDifferences(layers.size());

    // Output layer differences.
    int layerIndex = layers.size() - 1;
    networkDifferences[layerIndex] = vector<double>(layers[layerIndex].size());

    for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); ++neuronIndex) {
        // difference = expected - real
        networkDifferences[layerIndex][neuronIndex] = expectedOutputs[neuronIndex] - allResults[layerIndex][neuronIndex];
    }

    --layerIndex;

    // Hidden layer differences.

    while (layerIndex >= 0) {
        networkDifferences[layerIndex] = vector<double>(layers[layerIndex].size());

        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); ++neuronIndex) {
            // difference = sum(difference of next neuron j * weight of this neuron on next neuron j)
            networkDifferences[layerIndex][neuronIndex] = 0;

            for (int nextNeuronIndex = 0; nextNeuronIndex < layers[layerIndex + 1].size(); ++nextNeuronIndex) {
                networkDifferences[layerIndex][neuronIndex] += networkDifferences[layerIndex + 1][nextNeuronIndex] * layers[layerIndex + 1][nextNeuronIndex][neuronIndex];
            }
        }

        --layerIndex;
    }

    // Calculating deltas.
    vector<vector<vector<double> > > networkDeltas(layers.size());

    for (layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        networkDeltas[layerIndex] = vector<vector<double> >(layers[layerIndex].size());

        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); ++neuronIndex) {
            networkDeltas[layerIndex][neuronIndex] = vector<double>(layers[layerIndex][neuronIndex].size());

            for (int weightIndex = 0; weightIndex < layers[layerIndex][neuronIndex].size(); ++weightIndex) {
                // delta = trainingSpeed * difference assigned to this neuron * f'(sum(original inputs * original weights)) * inputs[weightIndex] of this neuron
                networkDeltas[layerIndex][neuronIndex][weightIndex] = trainingSpeed * networkDifferences[layerIndex][neuronIndex] * computeNeuron(allInputs[layerIndex], layers[layerIndex][neuronIndex], true) * allInputs[layerIndex][weightIndex];
            }
        }
    }

    // Calculating new weights.

    for (layerIndex = 0; layerIndex < layers.size(); ++layerIndex) {
        for (int neuronIndex = 0; neuronIndex < layers[layerIndex].size(); ++neuronIndex) {
            for (int weightIndex = 0; weightIndex < layers[layerIndex][neuronIndex].size(); ++weightIndex) {
                // newWeight = oldWeight + delta + lastDelta * momentum.
                layers[layerIndex][neuronIndex][weightIndex] += networkDeltas[layerIndex][neuronIndex][weightIndex];

                if (lastDeltas.size() > 0) {
                    layers[layerIndex][neuronIndex][weightIndex] += lastDeltas[layerIndex][neuronIndex][weightIndex] * momentum;
                }
            }
        }
    }

    lastDeltas = networkDeltas;
}
