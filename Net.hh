#ifndef NET_HH
#define NET_HH

#include <vector>
#include <random>
#include <math.h>

using namespace std;

class Net {
private:
    // Vector of layers, every layer is vector of neurons, every neuron is vector of weights. Matrix of 3 dimensions as result.
    vector<vector<vector<double> > > layers;

    // Vector of layers, every layer is vector of neurons, every neuron has 1 result.
    vector<vector<double> > allResults;
    vector<vector<double> > allInputs;

    void createNetwork(int inputSize, int hiddenSize, int outputSize);
    void createLayer(int size, int inputSize, int layerIndex);
    double getRandomWeight(int numberOfWeights);
    double computeNeuron(const vector<double> &inputs, const vector<double> &weights, bool derivative);
    double activationFunction(double x, bool derivative);

public:
    Net(int inputSize, int hiddenSize, int outputSize);

    void compute(const vector<double> &input, vector<double> &results);
    void train(const vector<double> &input, const vector<double> &expectedOutputs, double trainingSpeed);
};

#endif
