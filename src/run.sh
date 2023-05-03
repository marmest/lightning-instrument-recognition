#!/bin/bash
cd data
python3 data_preprocessing.py
cd ..
cd models
python3 train.py
cd ..
cd fusion
python3 fusion.py
