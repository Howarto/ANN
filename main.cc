#include <iostream>
#include "Neuron.hh"
#include "Net.hh"

using namespace std;

int main() {
    // INPUT EXAMPLE
    vector<double> input(2);
    input[0] = 0.6;
    input[1] = 0.35;

    Net net(3, 2, 1, input);
    vector<double> output(1);
    net.calculate_net(output);
    for (int i = 0; i < output.size(); ++i) {
        cout << output[i] << endl;
    }
    cout << endl;
    // CORRECT VALUE EXAMPLE
    vector<double> correct_value(1, 0.95);
    net.error(correct_value, output);

    net.propagation(0.9, input);

    net.calculate_net(output);
    for (int i = 0; i < output.size(); ++i) {
        cout << output[i] << endl;
    }
}

