#!/bin/bash

mamba create -n episegment python=3.10 -y
mamba activate episegment
mamba install -c pytorch -c nvidia -c conda-forge --file requirements.txt -y
