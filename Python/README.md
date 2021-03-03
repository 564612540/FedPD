# FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data

This repository is the official implementation of [FedPD: A Federated Learning Framework with Optimal Rates and Adaptivity to Non-IID Data](). 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training & Evaluation

To run the algorithms in the paper, run this command:

```train
python3 ./fed_main.py --epochs 1000 --freq_in 300 --freq_out 1 --num_users 90 --local_ep 300 --local_bs 2 --lr 0.01 --mu 1 --dataset femnist --VR False --model cnn --optimizer FedPD --gpu cuda:0
```
-optimizer can be selected from FedPD, FedProx, sgd and adam

## Results

Our algorithm achieves the following performance:

| Algorithm          | Train Accuracy  | Test Accuracy |
| ------------------ |---------------- | ------------- |
| FedPD              |       96%       |      89%      |
| FedProx            |       95%       |      87%      |
| FedAvg             |       91%       |      86%      |


## Contributing

Copyright (c) [2020] [Xinwei Zhang]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
