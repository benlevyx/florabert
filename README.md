# Bringing BERT to the field: Transformer models for gene expression prediction in maize
**Authors: Benjamin Levy, Shuying Ni, Zihao Xu, Liyang Zhao**  
Predicting gene expression levels from upstream promoter regions using deep learning. Collaboration between IACS and Inari.  

---
## Directory Setup
**`scripts/`: directory for production code**
- [`0-data-loading-processing/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/scripts/0-data-loading-processing):
    - [`01-gene-expression.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/01-gene-expression.py): downloads and processes gene expression data and saves into "B73_genex.txt".
    - [`02-download-process-db-data.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/02-download-process-db-data.py): downloads and processes gene sequences from a specified database: 'Ensembl', 'Maize', 'Maize_addition', 'Refseq'
    - [`03-combine-databases.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/03-combine-databases.py): combines all the downloaded sequences within all the databases
    - [`04a-merge-genex-maize_seq.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/04a-merge-genex-maize_seq.py):
    - [`04b-merge-genex-b73.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/04b-merge-genex-b73.py):
    - [`05a-cluster-maize_seq.sh`](scripts/0-data-loading-processing/05a-cluster-maize_seq.sh): clusters the promoter sequences into groups with up to 80% sequence identity, which may be interpreted as paralogs
    - [`05b-train-test-split.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/05-train-test-split.py): divides the promoter sequences into train and test sets, avoiding a set of pairs that indicate close relations ("paralogs")
    - [`06_transformer_preparation.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/06_transformer_preparation.py):
    - [`07_train_tokenizer.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/0-data-loading-processing/07_train_tokenizer.py): training byte-level BPE for RoBERTa model
- [`1-modeling/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/scripts/1-modeling)
    - [`pretrain.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/1-modeling/pretrain.py): training the CornBERT base using a masked language modeling task. Type `python scripts/1-modeling/pretrain.py --help` to see command line options, including choice of dataset and whether to warmstart from a partially trained model. Note: not all options will be used by this script.
    - [`finetune.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/1-modeling/finetune.py): training the CornBERT regression model (including newly initialized regression head) on multitask regression for gene expression in all 10 tissues. Type `python scripts/1-modeling/finetune.py --help` to see command line options; mainly for specifying data inputs and output directory for saving model weights.
    - [`evaluate.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/1-modeling/evaluate.py): computing metrics for the trained CornBERT model
- [`2-feature-visualization/](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/scripts/2-feature-visualization)`
    - [`embedding_vis.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/scripts/2-feature-visualization/embedding_vis.py): computing a sample of BERT embeddings for the testing data and saving to a tensorboard log. Can specify how many embeddings to sample with `--num-embeddings XX` where `XX` is the number of embeddings (must be integer).


**`module/`: directory for our customized modules**
- [`inari/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/module/inari): our main module named `inari` that packages customized functions
    - [`config.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/config.py): project-wide configuration settings and absolute paths to important directories/files
    - [`dataio.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/dataio.py): utilities for performing I/O operations (reading and writing to/from files)
    - [`gene_db_io.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/gene_db_io.py): helper functions to download and process gene sequences
    - [`metrics.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/metrics.py): functions for evaluating models
    - [`nlp.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/nlp.py): custom classes and functions for working with text/sequences
    - [`training.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/training.py): helper functions that make it easier to train models in PyTorch and with Huggingface's Trainer API, as well as custom optimizers and schedulers
    - [`transformers.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/transformers.py): implementation of RoBERTa model with mean-pooling of final token embeddings, as well as functions for loading and working with Huggingface's transformers library
    - [`utils.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/utils.py): General-purpose functions and code
    - [`visualization.py`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/module/inari/visualization.py): helper functions to perform random k-mer flip during data processing and make model prediction
    
**`notebooks/`: directory for jupyter notebooks**
- [`eda/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/notebooks/eda): exploratory data analysis
    - [`00_sequence_eda.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/eda/00_sequence_eda.ipynb): explores gene sequences, especially the distributions of neucleotides
    - [`01_gene_expression.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/eda/01_gene_expression.ipynb): performs exploratory data analysis on gene expression data.
    - [`02_diagonoise_new_genX_matrix.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/eda/02_diagnoise_new_genX_matrix.ipynb): a quick diagonoise on gene expression data (heatmap).

- [`language/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/notebooks/language):
- [`modeling/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/notebooks/modeling):
    - [`03_hand_eng_baseline.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/modeling/03_hand_eng_baseline.ipynb): the baseline model that uses hand-engineered features derived from a list of common plant promoters
    - [`05_baseline_bow.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/modeling/05_baseline_bow.ipynb): the baseline model that extracts bag-of-nucleotide from DNA sequence as features and performs Ridge Rregression models on top of it.
- [`process_data/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/notebooks/process_data):
    - [`00-upstream_sequence_extraction_original.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/process_data/00-upstream_sequence_extraction_original.ipynb): original notebook provided by Inari used for sequence extraction
- [`transformer/`](https://github.com/InariCapstone2020/inari-deep-gene/tree/master/notebooks/transformer):
    - [`00_language_model.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/transformer/00_language_model.ipynb): exploring the CornBERT language model and its performance on the masked language modeling task
    - [`01_genex_prediction.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/transformer/01_genex_prediction.ipynb): exploring the CornBERT regression model and how its predictions compare with the ground truth gene expression values for each tissue
    - `02_training_vis.ipynb`: visualizing the train/eval loss and progressive unfreezing process in training
    - [`05_Viz_kmer.ipynb`](https://github.com/InariCapstone2020/inari-deep-gene/blob/master/notebooks/transformer/05_Viz_kmer.ipynb): notebook that plots the importance graph to illustrate positions on the regulatory sequences where the CornBert model places more weights on.

### Pretrained models

If you wish to experiment with our pre-trained CornBERT models, you can find the saved PyTorch models and the Huggingface tokenizer files [here](https://drive.google.com/drive/folders/1qHwRfXxPVC1j2GcZ-wFOT3BmTmHRr_it?usp=sharing)

**Contents**:

- `byte-level-bpe-tokenizer`: Files expected by a Huggingface `transformers.PretrainedTokenizer`
    - `merges.txt`
    - `vocab.txt`
- transformer: Both language models can instantiate any RoBERTa model from Huggingface's `transformers` library. The prediction model should instantiate our custom `RobertaForSequenceClassificationMeanPool` model class
    1. `language-model`: Trained on all plant promoter sequences
    2. `language-model-finetuned`: Further trained on just maize promoter sequences
    3. `prediction-model`: Fine-tuned on the multitask regression problem
