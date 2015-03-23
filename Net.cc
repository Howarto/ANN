#include "Net.hh"
#include <random>

// Funciones de apoyo

// Sirve para generar un número "aleatorio" (en realidad es lineal...)
double Net::fRand(double fMin, double fMax) {
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

// Constructoras
Net::Net(int input_size, int hidden_size, int output_size, vector<double> &inp) {
    input = inp;
    input_layer = vector<Neuron>(input_size);
    hidden_layer = vector<Neuron>(hidden_size);
    output_layer = vector<Neuron>(output_size);

    // Initialize random weights
    for (int i = 0; i < input_size; ++i) {
        input_layer[i].length_weight_vector(inp.size());
        for (int j = 0; j < inp.size(); ++j) {
            // TENGO QUE GENERAR AQUÍ UN NÚMERO ALEATORIO!!
            input_layer[i].modify_i(j, fRand(-1.0,1.0));
        }
    }


//    // ESTO ES PARA EL EXAMPLE!!!
//    input_layer[0].modify_i(0, 1);
//    input_layer[0].modify_i(1, 0.15);

//    input_layer[1].modify_i(0, 0.5);
//    input_layer[1].modify_i(1, 0.75);

//    input_layer[2].modify_i(0, 0.8);
//    input_layer[2].modify_i(1, 1.2);

    for (int i = 0; i < hidden_size; ++i) {
        hidden_layer[i].length_weight_vector(input_size);
        for (int j = 0; j < input_size; ++j) {
            hidden_layer[i].modify_i(j, fRand(-1.0,1.5));
        }
    }

//    // ESTO ES PARA EL EXAMPLE!!!
//    hidden_layer[0].modify_i(0, 0.2);
//    hidden_layer[0].modify_i(1, 0.21);
//    hidden_layer[0].modify_i(2, 0.62);

//    hidden_layer[1].modify_i(0, 0.17);
//    hidden_layer[1].modify_i(1, 0.55);
//    hidden_layer[1].modify_i(2, 0.81);

    for (int i = 0; i < output_size; ++i) {
        output_layer[i].length_weight_vector(hidden_size);
        for (int j = 0; j < hidden_size; ++j) {
            output_layer[i].modify_i(j, fRand(-1.1,0.9));
        }
    }

//    // ESTO ES PARA EL EXAMPLE!!!
//    output_layer[0].modify_i(0, 0.09);
//    output_layer[0].modify_i(1, 0.4);

}

// Modificadores


void Net::calculate_net(vector<double> &output_outputlayer) {
    // Operaciones input layer
    vector<double> output_inputlayer(input_layer.size());
    for (int i = 0; i < input_layer.size(); ++i) {
        input_layer[i].calculate_net(input);
        output_inputlayer[i] = input_layer[i].calculate_output();
    }

    // Cuando llegue hasta aquí en output_inputlayer estarán todas
    // las salidas del primer layer.
    /* Ahora lo redirigiré hacia el hidden layer */

    vector<double> output_hiddenlayer(hidden_layer.size());
    for (int i = 0; i < hidden_layer.size(); ++i) {
        hidden_layer[i].calculate_net(output_inputlayer);
        output_hiddenlayer[i] = hidden_layer[i].calculate_output();
    }

    // Hice lo mismo que antes pero con la capa hidden. Ahora redigiré
    // hacia la última capa lo que queda.

    for (int i = 0; i < output_layer.size(); ++i) {
        output_layer[i].calculate_net(output_hiddenlayer);
        output_outputlayer[i] = output_layer[i].calculate_output();
    }

}

void Net::error(const vector<double> &correct_value, const vector<double> &output_outputlayer) {
    // Para cada salida hay que realizar el cálculo, ya que el error
    // total viene siendo el sumatorio de los errores. Cada uno se saca
    // de los outputs y se sacan los errores de cada neurona haciendo
    // un "barrido" hacia el input, calculando en cadena todas las neuronas
    // que hay por el camino.

    // Bucle for principal que recorrer los output
    for (int i = 0; i < output_layer.size(); ++i) {
        output_layer[i].modify_delta_value(correct_value[i] - output_outputlayer[i]);
    }

    // Recuerda: Las deltas de las capas anteriores al output siguen la fórmula
    // delta = sumatorio de (los pesos relaciones con la neurona de la capa que queremos calcular*delta capa siguiente).
    double aux;
    // Bucle para el hidden
    for (int i = 0; i < hidden_layer.size(); ++i) {
        aux = 0;
        for (int l = 0; l < output_layer.size(); ++l) {
            aux += output_layer[l].i_weight_vector(i) * output_layer[l].delta_value();
        }
        hidden_layer[i].modify_delta_value(aux);
    }

    // Bucle para el input
    for (int i = 0; i < input_layer.size(); ++i) {
        aux = 0;
        for (int l = 0; l < hidden_layer.size(); ++l) {
            aux += hidden_layer[l].i_weight_vector(i) * hidden_layer[l].delta_value();

        }
        input_layer[i].modify_delta_value(aux);
    }
}

void Net::propagation(double n, vector<double> &input) {
    // Input layer
    for (int i = 0; i < input_layer.size(); ++i) {
        for (int j = 0; j < input_layer[i].length(); ++j) {
            input_layer[i].modify_i(j,
                                  input_layer[i].i_weight_vector(j) +
                                  n*input_layer[i].delta_value()*
                                  funcion_derivada(input_layer[i].net_value())*input[j]);
        }
    }


    // Hidden layer

    for (int i = 0; i < hidden_layer.size(); ++i) {
        for (int j = 0; j < hidden_layer[i].length(); ++j) {
            hidden_layer[i].modify_i(j,
                        hidden_layer[i].i_weight_vector(j) +
                        n*hidden_layer[i].delta_value()*funcion_derivada(hidden_layer[i].net_value())*funcion(input_layer[j].net_value()));
        }
    }

    // Output layer
    for (int i = 0; i < output_layer.size(); ++i) {
        for (int j = 0; j < output_layer[i].length(); ++j) {
            output_layer[i].modify_i(j,
                        output_layer[i].i_weight_vector(j) +
                        n*output_layer[i].delta_value()*funcion_derivada(output_layer[i].net_value())*funcion(hidden_layer[j].net_value()));
        }
    }
}
