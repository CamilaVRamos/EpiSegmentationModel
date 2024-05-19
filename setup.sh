#!/bin/bash

mamba create -n project python=3.10 -y
mamba activate project
mamba install -c pytorch -c nvidia -c conda-forge --file requirements.txt -y
