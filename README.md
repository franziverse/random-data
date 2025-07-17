# Random Data

This project explores how deep learning models perform when training on different permutations of data across multiple datasets (MNIST1D, MNIST and CIFAR10). The permutations partially or completely deconstruct the underlying structure or regularities in the data.  

The goal is to compare how well the models can still learn and fit the data and what this means for the ability to generalize.

## Quickstart
**Clone the repository**  

```
git clone https://github.com/franziverse/random-data.git
cd random-data
```

**Set up the environment**  

Using conda  

```
conda env create -f env.yml
conda activate random_data
```

Using pip install  

```
pip install -r requirements.txt
```

**Open the notebook**  

```
jupyter lab
```

 
## Acknowledgements

This project uses code and ideas from:

- original notebook: '20_1_Random_Data.ipynb' from 'Understanding Deep Learning' by Simon Prince
- paper: 'UNDERSTANDING DEEP LEARNING REQUIRES RE-THINKING GENERALIZATION' by Zhang et. al
- MNIST1D dataset: 'mnist1d Git repro' by Samuel Greydanus

Original code was altered.


## Our contribution:

- Instead of using only one dataset we added the option to choose between three different datasets: MNIST1D, MNIST or CIFAR10 (MNIST1D was used in the original notebook, CIFAR10 was used in the paper)
- The original notebook contained permutations for randomized data (shuffled pixel) and permutated labels (shuffled labels), we also included random pixels, gaussian noise and gaussian noise overlying the original as permutation options (as is done in the original paper, so not a new idea but an addition to the original notebook)
- We added visualization for the data (original and permuted) and plots for visualizing the training
- Additionally to recording the loss over the cource of the training, the accuracy also is recorded and later compared as well
- Generalization is tested by running the model with test data and recording loss and accuracy
