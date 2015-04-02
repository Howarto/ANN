TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt
QMAKE_CXXFLAGS += -D_GLIBCXX_DEBUG -std=c++11
QMAKE_CXXFLAGS -= -Wsign-compare
HEADERS += *.hh
SOURCES += *.cc
include(deployment.pri)
qtcAddDeployment()

