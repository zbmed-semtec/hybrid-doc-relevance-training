**Note that in this directory, embeddings are generated for MeSHIDs with prefixes meshd/meshq/meshc/meshu, i.e. all articles' tokens are in lowercase.**

## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using one of the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/hybrid-post-word2doc2vec-doc-relevance-training.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/hybrid-post-word2doc2vec-doc-relevance-training.git
```

#### Navigate to the `training` Folder:

```
cd hybrid-post-word2doc2vec-doc-relevance-training/training
```

### Step 2: Create a virtual environment and install dependencies

To create a virtual environment within your repository, run the following command:

```
python3 -m venv .venv 
source .venv/bin/activate   # On Windows, use '.venv\Scripts\activate' 
```

To confirm if the virtual environment is activated and check the location of yourPython interpreter, run the following command:

```
which python    # On Windows command prompt, use 'where python'
                # On Windows PowerShell, use 'Get-Command python'
```
The code is stable with python 3.6 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Dataset
- Download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN)
- Keep the data in the below-specified format

![image](https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/assets/62026329/7b585c09-1fc3-4122-bf27-42c957de6edf)

### Step 4: Optimization Pipeline

This pipeline aims to optimize hyperparameters for hybrid post(reduction) Word-2Doc-2Vec model using Optuna, train the model with the optimal parameters, and evaluate its performance using three-class precision at 5 (Precision@5).

#### Pipeline Steps:

- **Hyperparameter Optimization**: Utilizes Optuna to search for the best hyperparameters for hybrid post(reduction) Word-2Doc-2Vec model.
- **Model Training**: Trains hybrid post(reduction) Word-2Doc-2Vec model with the optimal hyperparameters using 80% of the training split data.
- **Embedding Generation**: Generates embeddings for the remaining 20% of the test split data using the trained model.
- **Cosine Similarity Computation**: Calculates cosine similarities for the generated embeddings.
- **Precision@N Calculation**: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
- **NDCG Score Calculation**: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this [script](./code/main.py), and run the following command:

```
python3 code/main.py [-i TRAIN_DATA] [-v VALIDATION_DATA] [-anv ANNOTATED_VALIDATION_DATA] [-t TEST_DATA] [-ant ANNOTATED_TEST_DATA] [-gv VALIDATION_GROUND_TRUTH] [-gt TEST_GROUND_TRUTH] [-dict MeShIDtoPMID] [-rd REDUCTION_OR_NOT]
```

You must pass the following four arguments:

+ -i/ --input : File path to RELISH Train split dataset (.npy file format).
+ -v/ --valid :  File path to RELISH Validation split dataset (.npy file format).
+ -anv/ --Annot_valid: File path to RELISH Annotated Validation split dataset (.npy file format).
+ -t/ --test :  File path to the RELISH Test split dataset (.npy file format).
+ -ant/ --Annot_test: File path to RELISH Annotated Test split dataset (.npy file format).
+ -gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
+ -gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
+ -dict/ --MeShIDtoPMID : File path to input MeShIDtoPMID (.tsv file format).
+ -rd/ --reduction : Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0).

To run this script for hybrid-post-Word2Doc-2Vec, you may execute the following command:

```
python3 code/main.py -i relish_train_tokens_removed_stopwords.npy -v relish_val_tokens_removed_stopwords.npy -anv relish_val_annotated_tokens_removed_stopwords.npy -t relish_test_tokens_removed_stopwords.npy -ant relish_test_annotated_tokens_removed_stopwords.npy -gv val_split.tsv -gt test_split.tsv -dict ../data/dic_MeShIDtoPMID_2022628.tsv -rd 0
```
and in case of reduction, i.e. for hybrid-postreduction-Word2Doc-2Vec, you may execute the following command:
```
python3 code/main.py -i relish_train_tokens_removed_stopwords.npy -v relish_val_tokens_removed_stopwords.npy -anv relish_val_annotated_tokens_removed_stopwords.npy -t relish_test_tokens_removed_stopwords.npy -ant relish_test_annotated_tokens_removed_stopwords.npy -gv val_split.tsv -gt test_split.tsv -dict ../data/dic_MeShIDtoPMID_2022628.tsv -rd 1
```

All outputs of the [script](./code/main.py) are saved in the folder named `output_of_model`. During the optimization process, we can monitor the progress via `Optuna_trials.log` saved in the folder `output_of_model`. After completing the code run, all trials' information will be saved in `optuna_study_state.csv`.

Note that this [script](./code/main.py) creates a [resumable Optuna study](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#rdb). Specifically, if the optimization process is interrupted or stopped for any reason, or if there's a desire to continue the optimization process further after completing the code run, it's possible to resume the Optuna study that was previously created. To do so, all that's required is the file `optuna_study_storage.db`, which should be saved in the `output_of_model` folder, and then re-executing the [script](./code/main.py).
