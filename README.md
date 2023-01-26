# Cancer Immunotherapy Data Science Grand Challenge - Challenge 1
Username: jackson6

## Summary

My solution is an ensemble of eight identical multi-layer perceptrons (MLPs) built using Tensorflow writen in Python. \
To regenerate the submission state proportion files see the "Run Inference" section below. \
To generate new weights see the "Run Training" section below (this will overwrite the included pretrained weights). 

# Setup

0. Set the working directory as the 'code' directory
```
cd code
```

1. Create an environment using Python 3.8. The solution was originally run on Python 3.8.16. 
```
conda create --name cim-submission python=3.8
```

then activate the environment
```
conda activate cim-submission
```

2. Install the required Python packages:
```
pip install -r requirements.txt
```

(Optional) for GPU accelerated environments:

```
pip install tensorflow-gpu==2.9.2
```


The structure of the directory before running training or inference should be:
```
code
├── data
│   ├── processed      <- Output folder for training & inference
│   │   ├── model-0.h5
│   │   ├── model-1.h5
│   │   ...
│   ├── embeddings     
│   │   └──g2v.txt <- Genevector embeddings
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
                          [default: 7]

  --teacher-dim INTEGER   The hidden dimension of
                          the teacher (NOT used
                          for inference) MLP
                          [default: 128]

  --student-dim INTEGER   The hidden dimension of
                          the student (used for
                          inference) MLP
                          [default: 256]

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
By default, predictions will be saved out to `data/processed/validation_output.csv`. 


To run inference on the TEST genes: `python src/run_inference.py --test-mode` \
By default, predictions will be saved out to `data/processed/test_output.csv`.


```commandline
$python src/run_inference.py --help
Usage: run_inference.py [OPTIONS]

Options:
  --test-mode / --no-test-mode  Predict the test
                                genes, otherwise
                                predict the
                                validation genes
                                [default: False]

  --submission-save-dir PATH    Directory to save
                                csv  [default:
                                ./data/processed]

  --model-dir PATH              Directory to save
                                the output model
                                weights  [default:
                                ./data/processed]

  --n-models INTEGER            Number of models
                                to use in the
                                model-dir
                                [default: 7]

  --n-samples INTEGER           Number of
                                generated samples
                                [default: 1000]

  --help                        Show this message
                                and exit.
```
# Hardware

The solution was run on a Google colab notebook
- Number of CPUs: 4
- Processor: Intel(R) Xeon(R) CPU @ 2.20GHz
- Memory: 12 GB 
- GPU: Tesla T4

Both training and inference were run on GPU.
- Training time: ~ 45 mins
- Inference time: ~ 10 sec

