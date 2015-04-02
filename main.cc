#include <iostream>
#include "Neuron.hh"
#include "Net.hh"

using namespace std;

int main() {
    // INPUT EXAMPLE
//    vector<double> input(2);
//    input[0] = 0.6;
//    input[1] = 0.35;

    Trait sample;

    // Creo los diferentes samples
    sample.inputs = vector< vector<double> >(4, vector<double>(2));

    sample.inputs[0][0] = 0;
    sample.inputs[0][1] = 0;

    sample.inputs[1][0] = 0;
    sample.inputs[1][1] = 1;

    sample.inputs[2][0] = 1;
    sample.inputs[2][1] = 0;

    sample.inputs[3][0] = 1;
    sample.inputs[3][1] = 1;

    // Creo las correspondientes salidas válidas. Las creo en formato columna, es decir,
    // en cada fila estarán los outputs de cada vector de input.
    sample.g_val = vector< vector<double> >(4, vector<double>(1));
    sample.g_val[0][0] = 0;
    sample.g_val[1][0] = 0;
    sample.g_val[2][0] = 0;
    sample.g_val[3][0] = 1;


    Net net(2, 1, 1, sample.inputs[0]);
    vector <double> output_net(1);
    net.calculate_network_net(sample.inputs[0] ,output_net);
    int l = 0;
    bool seguir = true;
    bool b; // Indica false si ha tenido que entrar en el cálculo del error dentro de train()
    int cuenta = 0;
    int h = 0;
    while (h < 1000) {
        net.train(output_net, sample.g_val[l], 0.1, sample.inputs[l], b);
        if (not b) ++cuenta;
        if (cuenta == sample.inputs[0].size()) seguir = false;
        ++l;
        if (l >= 4) {
            l = 0;
            cuenta = 0;
        }
        ++h;
    }

    net.calculate_network_net(sample.inputs[0], output_net);
    cout << "Debería dar 0 " << output_net[0] << endl;

    net.calculate_network_net(sample.inputs[1], output_net);
    cout << "Debería dar 0 " << output_net[0] << endl;

    net.calculate_network_net(sample.inputs[2], output_net);
    cout << "Debería dar 0 " << output_net[0] << endl;

    net.calculate_network_net(sample.inputs[3], output_net);
    cout << "Debería dar 1 " << output_net[0] << endl;

}

