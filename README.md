# A Sub-Problem Quantum Alternating Operator Ansatz for Correlation Clustering

This repository contains the code for the experiments conducted in the paper [A Sub-Problem Quantum Alternating Operator Ansatz for Correlation Clustering](https://icml.cc/virtual/2025/poster/45107), accepted at ICML'25 .

## Installing
```shell
conda env create -f environment.yml
```

## Experiments
The notebook `benchmark_SQAOA.ipynb` performs the experiments thate measure the approximation ratios and runtimes for SQAOA on Erdős–Rényi graphs with three nodes.
The other experiments can be easily reproduced by adapting the parameters in the third cell.