# Random Data

This project explores how deep learning models perform when training on different corruptions of data across multiple datasets (MNIST1D, MNIST and CIFAR10). The corruptions partially or completely deconstruct the underlying structure or regularities in the data.  

The corruptions to choose from are:

- Shuffled labels: destroys the structure in the maping between the data and the labels
- Shuffled pixels: (rearangement of the pixels) destroys the structure within the pictures 
- Random pixels: randomly generated pixels (so no structure at all)
- Gaussian noise (general): generates pixels, not randomly but based on the gaussian distribution of the original dataset (only keeps structure statistically)
- Gaussian noise (individual): similar to the previous one but uses the gaussian distribution of each individual image instead
- Original + gaussian noise: Gaussian noise is superimposed on the original pictures (keeps the structure intact but adds noise)

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
- We added visualization for the data (original and corrupted) and plots for visualizing the training
- Additionally to recording the loss over the cource of the training, the accuracy also is recorded and later compared as well
- Generalization is tested by running the model with test data and recording loss and accuracy
