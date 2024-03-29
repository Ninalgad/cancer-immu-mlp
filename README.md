
# Cancer Immunotherapy Data Science Grand Challenge - Challenge 1 (8th Place Solution)

## Summary

My solution is an ensemble of eight identical multi-layer perceptrons (MLPs) built using Tensorflow written in Python. \
After training new weights with the "Run Training" section, the submission state proportion files can be regenerated with the "Run Inference" section. Both sections are below. \

# Setup

0. Clone and set the working directory as the 'cancer-immu-mlp' directory
```commandline
git clone https://github.com/Ninalgad/cancer-immu-mlp.git
cd code
```

1. Create an environment using Python 3.8. The solution was originally run on Python 3.8.16. 
```commandline
conda create --name cim-submission python=3.8
```

then activate the environment
```commandline
conda activate cim-submission
```

2. Install the required Python packages:
```commandline
pip install -r requirements.txt
```

(Optional) for GPU accelerated environments:

```commandline
pip install tensorflow-gpu==2.9.2
```


The structure of the directory before running training or inference should be:
```
cancer-immu-mlp
├── data
│   ├── processed      <- Output folder for training & inference
│   │   ├── model-0.h5
│   │   ├── model-1.h5
│   │   ...
│   ├── embeddings     
│   │   └── gene2vec_dim_200_iter_9_w2v.txt <- Genevector embeddings
│   └── raw            <- The original data files
│       ├── sc_training.h5ad
│       ├── clone_information.csv
│       ├── guide_abundance.csv
│       └── scRNA_ATAC.h5
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── make_dataset.py
│   ├── eval.py
│   ├── g2v.py
│   ├── run_inference.py
│   ├── model.py
│   ├── loss.py
│   ├── run_training.py
│   └── training_utils.py
├── README.md          <- The top-level README for using this project.
├── requirements.txt   <- List of all dependencies
├── Makefile           <- Makefile with commands like `make requirements`
└── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
```

# Run Training

To run training: `python src/run_training.py`. 

```commandline
$python src/run_training.py --help
Usage: run_training.py [OPTIONS]

Options:
  --model-dir PATH        Directory to save the
                          output model weights in
                          npy format  [default:
                          ./data/processed]

  --features-dir PATH     Path to the raw features
                          [default: ./data/raw/]

  --n-folds INTEGER       Number of folds/models,
                          Must be at least 2
                          [default: 5]

  --model-dim INTEGER     The hidden dimension of
                          the model  [default:
                          256]

  --random-state INTEGER  Controls the randomness
                          of each fold and noise
                          [default: 758625225]

  --debug / --no-debug    Run on a small subset of
                          the data and a two folds
                          for debugging  [default:
                          False]

  --help                  Show this message and
                          exit.
```


# Run Inference

To run inference on the VALIDATION genes: `python src/run_inference.py` \
By default, predictions will be saved to `data/processed/validation_output.csv`. 


To run inference on the TEST genes: `python src/run_inference.py --test-mode` \
By default, predictions will be saved to `data/processed/test_output.csv`.


```commandline
$python src/run_inference.py --help
Usage: run_inference.py [OPTIONS]

Options:
  --test-mode / --no-test-mode  Predict the test
                                genes, otherwise
                                predict the
                                validation genes
                                [default: False]

  --submission-save-dir PATH    Predict the test
                                genes, otherwise
                                predict the
                                validation genes
                                [default:
                                ./data/processed]

  --model-dir PATH              Directory to save
                                the output model
                                weights  [default:
                                ./data/processed]

  --n-models INTEGER            Number of models
                                to use in the
                                model-dir
                                [default: 5]

  --n-samples INTEGER           Path to the
                                Gene2Vec
                                embeddings
                                [default: 1000]

  --help                        Show this message
                                and exit.
```
# Hardware

The solution was run on a Google Colab notebook
- Number of CPUs: 4
- Processor: Intel(R) Xeon(R) CPU @ 2.20GHz
- Memory: 12 GB 
- GPU: Tesla T4

Both training and inference were run on GPU.
- Training time: ~ 25 mins
- Inference time: ~ 10 sec

# Gene2vec Embeddings
repo: [https://github.com/jingcheng-du/Gene2vec](https://github.com/jingcheng-du/Gene2vec) MIT License: [jingcheng-du/Gene2vec/LICENSE](jingcheng-du/Gene2vec/LICENSE)

Gene2vec (Du et al., 2018) is a set of high-dimensional embeddings of human genes, where embeddings are closer if they are often expressed together. Here we feed these embeddings alongside random noise to a deep neural network.

# References:
Du, J., Jia, P., Dai, Y., Tao, C., Zhao, Z., & Zhi, D. (2018). Gene2vec: distributed representation of genes based on co-expression. BMC Genomics, 20.
