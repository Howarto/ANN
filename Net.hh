#ifndef NET_HH
#define NET_HH

#include "Neuron.hh"

//  OJO!! QUE NO HE IMPLEMENTADO BIAS!!
class Net {
private:
    vector<double> input;
    vector<double> output;
    vector<Neuron> input_layer;
    vector<Neuron> hidden_layer;
    vector<Neuron> output_layer;

    static double fRand(double fMin, double fMax);


public:

    // Constructoras
    /** \brief Constructor. */
    Net(int input_size, int hidden_size, int output_size, vector<double> &input);

    // Modificadores

    /** \brief Take a net of the all network's layers.
     *  \pre All the sizes > 0 and the vector is not empty
     *  \post The vector is full of the output value and the net
     *   of the network is calculate */
    void calculate_net(vector<double> &output_outputlayer);

    /** \brief Take a error of the all net's neuron
     *  \pre The vectors are not empty
     *  \post It calculates delta value */
    void error(const vector<double> &correct_value, const vector<double> &output_outputlayer);

    /** \brief It propagates a changes of the weights
     *  \pre n > 0 and the vector is not empty
     *  \post All the weights are been recalculated */
    void propagation(double n, vector<double> &input);

    /** \brief It trains ANN
    void train();

};

#endif
