// #include <howarto>
#ifndef NEURON_HH
#define NEURON_HH

#include <iostream>
#include <vector>
#include "Funcion.hh"

using namespace std;

class Neuron {

private:
    double net;
    double delta;
    vector<double>weight;

public:


    // Constructores
    Neuron();

    // Modificadores

    void length_weight_vector(int x);

    /** \brief It calculates net value
        \pre two vector have minimum 1 element
        \post It modificates net with the new net value */
    void calculate_net(vector<double> &input);

    /** \brief It calculates output value.
        \pre true.
        \post It returns output value. */
    double calculate_output();

    /** \brief Modify i position in weight vector.
     *  \pre i is a valid vector index.
     *  \post "i" position in weight vector is modified. */
    void modify_i(int i, double value);

    void modify_delta_value(double x);

    // Consultores

    double delta_value() const;

    double net_value() const;

    /** \brief Return a value of "i" position in weight vector.
     *  \pre i = [0..input_number - 1]
     *  \post return double value */
    double i_weight_vector(int i) const;

    /** \brief Return a weight length
     *  \pre True.
     *  \post return weight length */
    int length() const;

};

#endif