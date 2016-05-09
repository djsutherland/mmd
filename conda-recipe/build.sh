#!/bin/bash

if [[ `uname` == Darwin ]]; then
    # make sure we're using conda gcc
    export CC=gcc
    export CXX=g++
    export LD=gcc
fi

$PYTHON setup.py egg_info
sed -n -e 's/^Version: \(.*\)$/\1/p' < mmd.egg-info/PKG-INFO > __conda_version__.txt

$PYTHON setup.py install
