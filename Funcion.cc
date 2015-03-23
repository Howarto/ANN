// #include <howarto>
#include "Funcion.hh"

double funcion(double x) {
    return ((tanh(x) + 1)/2);
}

double funcion_derivada(double x) {
    return ((1 - pow(tanh(x), 2))/2);
}
