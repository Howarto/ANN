#include <iostream>
#include "Net.hh"
#include "math.h"
#include <string>

using namespace std;

#define TRAINING_SPEED 0.9
#define INPUT_SIZE 2
#define HIDDEN_SIZE 2
#define OUTPUT_SIZE 1
#define TRAINING_CASES 4

#define DEBUG false

string v2s(const vector<double> &v) {
    string s = "";

    for (int i = 0; i < v.size(); ++i) {
        s = s + std::to_string(v[i]);

        if (i < v.size() - 1) {
            s += ", ";
        }
    }

    return s;
}

int main() {
    vector<vector<double> > inputs = vector<vector<double> >(TRAINING_CASES, vector<double>(INPUT_SIZE));
    vector<vector<double> > outputs = vector<vector<double> >(TRAINING_CASES, vector<double>(OUTPUT_SIZE));

    // AND
    /*
    inputs[0][0] = -1;
    inputs[0][1] = -1;

    inputs[1][0] = -1;
    inputs[1][1] = 1;

    inputs[2][0] = 1;
    inputs[2][1] = -1;

    inputs[3][0] = 1;
    inputs[3][1] = 1;

    outputs[0][0] = 0;
    outputs[1][0] = 0;
    outputs[2][0] = 0;
    outputs[3][0] = 1;
    */

    // OR
    //*
    inputs[0][0] = -1;
    inputs[0][1] = -1;

    inputs[1][0] = -1;
    inputs[1][1] = 1;

    inputs[2][0] = 1;
    inputs[2][1] = -1;

    inputs[3][0] = 1;
    inputs[3][1] = 1;

    outputs[0][0] = 0;
    outputs[1][0] = 1;
    outputs[2][0] = 1;
    outputs[3][0] = 1;
    //*/

    // XOR
    //*
    inputs[0][0] = -1;
    inputs[0][1] = -1;

    inputs[1][0] = -1;
    inputs[1][1] = 1;

    inputs[2][0] = 1;
    inputs[2][1] = -1;

    inputs[3][0] = 1;
    inputs[3][1] = 1;

    outputs[0][0] = 0;
    outputs[1][0] = 1;
    outputs[2][0] = 1;
    outputs[3][0] = 0;
    //*/

    Net net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE);
    bool finished = false;
    vector<double> results;
    int iteration = 1;

    while (!finished) {
        cout << "Iteration " << iteration << endl;

        for (int i = 0; i < TRAINING_CASES; ++i) {
            if (DEBUG) cout << "Training " << v2s(inputs[i]) << " => " << v2s(outputs[i]) << endl;
            net.train(inputs[i], outputs[i], TRAINING_SPEED);
        }

        finished = true;

        for (int i = 0; i < TRAINING_CASES && finished; ++i) {
            net.compute(inputs[i], results);
            if (DEBUG) cout << "Checking " << v2s(inputs[i]) << " -> " << v2s(results) << endl;
            finished = (round(results[0]) == outputs[i][0]);
            if (DEBUG) cout << v2s(results) << " == " << v2s(outputs[i]) << " ? " << finished << endl;
        }

        if (DEBUG) {
            cout << "Press enter to continue.";
            cin.ignore();
        }

        ++iteration;
    }

    for (int i = 0; i < TRAINING_CASES; ++i) {
        net.compute(inputs[i], results);
        cout << "Debería dar " << outputs[i][0] << " y da " << round(results[0]) << endl;
    }
}

