#!/bin/bash

if [[ `uname` == Darwin ]]; then
    # make sure we're using conda gcc
    export CC=gcc
    export CXX=g++
    export LD=gcc
fi

$PYTHON setup.py install
