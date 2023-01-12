# Cancer Immunotherapy Data Science Grand Challenge - Challenge 1
Username: jackson6

## Summary

My solution is an ensemble of ten identical multi-layer perceptrons (MLPs) built using Tensorflow writen in Python.
The final submission was produced by first training the model models produced by \
`python src/run_training.py`, then running `python src/run_inference.py` and `python src/run_inference.py --test-mode` to produce the validation and test csv(s) respectively.


# Setup

0. Change the working directory
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

3. Download and place the data from the competition page into `data/raw`


then download the Gene2vec embeddings with 
```
pip install wget
wget -O data/embeddings/gene2vec_dim_200_iter_9_w2v.txt https://github.com/jingcheng-du/Gene2vec/raw/master/pre_trained_emb/gene2vec_dim_200_iter_9_w2v.txt
```


The structure of the directory before running training or inference should be:
```
code
├── data
│   ├── processed      <- Output of training & inference
│   ├── embeddings     <- Gene2vec embeddings
│   │   └── gene2vec_dim_200_iter_9_w2v.txt
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

# Run training

To run training: `python src/run_training.py`. 


# Run inference

To run inference on the VALIDATION genes: `python src/run_inference.py` \
By default, predictions will be saved out to `data/processed/validation_output.csv`. 


To run inference on the TEST genes: `python src/run_inference.py --test-mode` \
By default, predictions will be saved out to `data/processed/test_output.csv`.


# Hardware

The solution was run on a Google colab notebook
- Number of CPUs: 4
- Processor: Intel(R) Xeon(R) CPU @ 2.20GHz
- Memory: 12 GB 
- GPU: Tesla T4

Both training and inference were run on GPU.
- Training time: ~ 12 mins
- Inference time: ~ 10 sec

# Gene2vec Embeddings
repo: [https://github.com/jingcheng-du/Gene2vec](https://github.com/jingcheng-du/Gene2vec)
MIT License: [jingcheng-du/Gene2vec/LICENSE](https://github.com/jingcheng-du/Gene2vec/blob/master/LICENSE)
<p>Gene2vec (Du et al., 2018) is a set of high-dimensional embeddings of human genes, where 
embeddings are closer together if they are often expressed together. Here we feed these embedding alongside random noise
to a deep neural network.</p>


--------
# References:
Du, J., Jia, P., Dai, Y., Tao, C., Zhao, Z., & Zhi, D. (2018). Gene2vec: distributed representation of genes based on co-expression. BMC Genomics, 20.

