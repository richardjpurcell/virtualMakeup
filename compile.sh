#!/bin/bash

cd /home/rpurcell/Documents/virtualMakeup/build
cmake ..
make
mv submission ..
cd ..
./submission