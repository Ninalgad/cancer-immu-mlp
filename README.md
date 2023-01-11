# Cancer Immunotherapy Data Science Grand Challenge - Challenge 1
Username: jackson6

## Summary

My solution is an ensemble of ten identical Deep Neural Networks (DNNs) built using tensorflow. 

The final submission was produced by ensembling the models produced by `src/run_kfold_training.py`, which trains on both the `train` and `val` splits specified in `raw/metadata.csv`.

# Setup

0. Clone (or unzip) and change the directory
```
git clone https://github.com/Ninalgad/cancer-immu-mlp.git
cd cancer-immu-mlp
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

3. Download and place the data from the competition page into `data/raw`


then download the Gene2vec embeddings with 
```
wget -O data/embeddings/gene2vec_dim_200_iter_9_w2v.txt https://github.com/jingcheng-du/Gene2vec/raw/master/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt
```


The structure of the directory before running training or inference should be:
```
cancer-immu-mlp
├── data
│   ├── processed      <- Output of training
│   ├── embeddings     <- Gene2vec embeddings
│   │   └── gene2vec_dim_200_iter_9_w2v.txt
│   └── raw            <- The original data files
│       ├── sc_training.h5ad
│       ├── clone_information.csv
│       ├── guide_abundance.csv
│       └── scRNA_ATAC.h5
├── models             <- Pre-trained model weights in h5 format
│   ├── s0.h5
│   ├── s1.h5
│   ...
├── src                <- Source code for use in this project.
│   ├── __init__.py    <- Makes src a Python module
│   ├── make_dataset.py
│   ├── run_inference.py
│   ├── model.py
│   ├── loss.py
│   ├── run_kfold_training.py
│   ├── training_utils.py
│   └── run_single_training.py
├── README.md          <- The top-level README for using this project.
├── requirements.txt   <- List of all dependencies
├── Makefile           <- Makefile with commands like `make requirements`
└── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
```

# Hardware

The solution was run on a Google colab notebook
- Number of CPUs: 4
- Processor: Intel(R) Xeon(R) CPU @ 2.20GHz
- Memory: 12 GB 
- GPU: Tesla T4

Both training and inference were run on GPU.
- Training time: ~ 6 hours
- Inference time: ~ 15 mins

# Run training

To run training using from the command line: `python src/run_kfold_training.py`. 

```
$ python src/run_kfold_training.py --help
```


# Run inference

Trained model weights can be downloaded from this Google folder: https://drive.google.com/drive/folders/1ujIuxB5R62ik-5-5wg9gp5uvmR8qh57z?usp=sharing

Ensure the weights are located in the `models` folder.


To run inference from the command line: `python src/run_inference.py`

```
$ python src/run_inference.py --help
```

By default, predictions will be saved out to `data/processed/submission.csv`.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>