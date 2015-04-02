// #include <howarto>
#include "Funcion.hh"

double funcion(double x) {
    return (1 / (1 + exp(-x)));
}

double funcion_derivada(double x) {
    return (-(funcion(x) * (1 - funcion(x))));
}
