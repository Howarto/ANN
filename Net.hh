#ifndef NET_HH
#define NET_HH

#include "Neuron.hh"

struct Trait {
    vector< vector<double> > inputs;    // Inputs
    vector< vector<double> > g_val;   // Good values
};

//  OJO!! QUE NO HE IMPLEMENTADO BIAS!!
class Net {
private:
    vector<Neuron> input_layer;
    vector<Neuron> hidden_layer;
    vector<Neuron> output_layer;

    static double fRand(double fMin, double fMax);


public:

    // Constructoras
    /** \brief Constructor. */
    Net(int input_size, int hidden_size, int output_size, vector<double> &inp);

    // Modificadores

    /** \brief Take a net of the all network's layers.
     *  \pre All the sizes > 0 and the vector is not empty
     *  \post The vector is full of the output value and the net
     *   of the network is calculate */
    void calculate_network_net(const vector<double> &input, vector<double> &output_outputlayer);

    /** \brief Take a error of the all net's neuron
     *  \pre The vectors are not empty
     *  \post It calculates delta value */
    void error(const vector<double> &correct_value, const vector<double> &output_outputlayer);

    /** \brief It propagates a changes of the weights
     *  \pre n > 0 and the vector is not empty
     *  \post All the weights are been recalculated */
    void propagation(double n, const vector<double> &input);

    // COMENTA!!
    bool interpretate(const vector<double> &correct_value,
                      vector<double> &output_outputlayer);

    /** \brief It trains ANN with specific values.
     *  \pre The vectors are not empty, n and error_admited != 0 and it
     *  must have a correct net value before use it.
     *  \post It has do one epoch, ANN's weights have been modificated
     *  one time and the new net value has been calculated */
    void train(vector<double> &output_outputlayer,
               const vector<double> &correct_value,
               double n,
               const vector<double> &input, bool &b);

};

#endif
