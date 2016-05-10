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

if [[ `uname` == Darwin ]]; then
    dir=($PREFIX/lib/python$PY_VER/site-packages/mmd-*.egg/mmd)
    if [[ "${#dir[@]}" -ne 1 ]]; then
        echo "Wrong number of matching directories..."
        echo "'${dir[@]}'"
        exit 2
    fi
    libgomp=$(otool -L $dir/_mmd.so | grep -Eo '[^[:space:]]*libgomp[^[:space:]]*')
    install_name_tool -change "$libgomp" "$PREFIX/lib/$(basename $libgomp)" "$dir/_mmd.so"
fi
