// #include <howarto>
#include "Neuron.hh"

// Constructores

Neuron::Neuron(){};

// Consultores

double Neuron::i_weight_vector(int i) const{
    return weight[i];
}

int Neuron::length() const {
    return (weight.size());
}

double Neuron::delta_value() const {
    return delta;
}

double Neuron::net_value() const {
    return net;
}

// Modificadores

void Neuron::modify_delta_value(double x) {
    delta = x;
}

void Neuron::length_weight_vector(int x) {
    weight = vector<double>(x);
}

void Neuron::calculate_net(vector<double> &input) {
    net = 0;
    for (int i = 0; i < input.size(); ++i) {
        net += input[i]*weight[i];
    }
}

double Neuron::calculate_output() {
     return funcion(net);
}

void Neuron::modify_i(int i, double value) {
    weight[i] = value;
}

