**Note that in this directory, models are trained and embeddings are generated for data with Removed-Stopwords**

In order to generalize the code for data including Stopwords, some changes are required in script [utilities.py](./code/utilities.py). Specifically, `pattern_to_find = [w for w in pmid_term[1:] if not w in stop_words]` in line 151 of function `generate_post_npy_dict_via_injection_MeSHIDs_into_tokens`, and line 311 of function `replacement_of_MeSHterms_with_MeSHIDs_in_tokens` must be replaced with `pattern_to_find = pmid_term[1:]`.

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

#### Navigate to the `training-PostReduction-WMD` Folder:

```
cd hybrid-post-word2doc2vec-doc-relevance-training/training-PostReduction-WMD
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

This pipeline aims to optimize hyperparameters of Word2Vec model for hybrid post WMD Word-2Doc-2Vec approach using Optuna, train the model with the optimal parameters, and evaluate its performance using three-class precision at 5 (Precision@5).

#### Pipeline Steps:

- **Hyperparameter Optimization**: Utilizes Optuna to search for the best hyperparameters of Word2Vec model within hybrid post WMD Word-2Doc-2Vec approach.
- **Model Training**: Trains Word2Vec model for hybrid post WMD Word-2Doc-2Vec approach with the optimal hyperparameters using the training split data.
- **Similarity Computation**: Calculates [WMD (Word Mover’s Distance)](../docs/WMD.md) similarities for the validation dataset using Gensim implementation of WMD, which utilizes the trained Word2Vec model.
- **Precision@N Calculation**: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained WMD similarities.
- **NDCG Score Calculation**: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this [script](./code/main.py), and run the following command:

```
python3 code/main.py [-i TRAIN_DATA] [-v VALIDATION_DATA] [-t TEST_DATA] [-gv VALIDATION_GROUND_TRUTH] [-gt TEST_GROUND_TRUTH] [-dict MeShIDtoPMID] [-win USING_WINDOWS]
```

You must pass the following four arguments:

+ -i/ --input : File path to RELISH Train split dataset (.npy file format).
+ -v/ --valid :  File path to RELISH Validation split dataset (.npy file format).
+ -t/ --test :  File path to the RELISH Test split dataset (.npy file format).
+ -gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
+ -gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
+ -dict/ --MeShIDtoPMID : File path to input MeShIDtoPMID (.tsv file format).
+ -win/ --windows : Assign 1 if using Windows systems; and 0 if using Unix-like systems (including Ubuntu).

For instance, to run this script on Windows systems, you may execute the following command:

```
python3 code/main.py -i relish_train_tokens_removed_stopwords.npy -v relish_val_tokens_removed_stopwords.npy -t relish_test_tokens_removed_stopwords.npy -gv val_split.tsv -gt test_split.tsv -dict ../data/dic_MeShIDtoPMID_2022628.tsv -win 1
```
and when using Unix-like systems (including Ubuntu), you may execute the following command:
```
python3 code/main.py -i relish_train_tokens_removed_stopwords.npy -v relish_val_tokens_removed_stopwords.npy -t relish_test_tokens_removed_stopwords.npy -gv val_split.tsv -gt test_split.tsv -dict ../data/dic_MeShIDtoPMID_2022628.tsv -win 0
```

All outputs of the [script](./code/main.py) are saved in the folder named `output_of_model`. During the optimization process, we can monitor the progress via `Optuna_trials.log` saved in the folder `output_of_model`. After completing the code run, all trials' information will be saved in `optuna_study_state.csv`.

Note that this [script](./code/main.py) creates a [resumable Optuna study](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html#rdb). Specifically, if the optimization process is interrupted or stopped for any reason, or if there's a desire to continue the optimization process further after completing the code run, it's possible to resume the Optuna study that was previously created. To do so, all that's required is the file `optuna_study_storage.db`, which should be saved in the `output_of_model` folder, and then re-executing the [script](./code/main.py).

Also, the best validation's trained model and its corresponding embeddings and cosine similarities are saved by the code in file-paths `output_of_model/model/best_Word2Vec_model`, `output_of_model/doc_embeddings/best_embeddings_pickle.pkl` and `output_of_model/evaluation/best_cosine_similarity.tsv`, correspondingly. This could be helpful to avoid redundant calculations when using the same dataset for both validation and testing. The saved best trained model from the validation phase is then loaded during the test phase to perform evaluation using test data.

Unfortunately, the use of `model.wv.add_vectors` to add new MeSHembedding-vectors in batches in the function `injection_MeSHembeddings_into_embeddings` of the script [utilities.py](./code/utilities.py) renders the saved best model from the tuning phase unloading. Therefore, to add new MeSHembedding-vectors, `model.wv.add_vector()` is utilized,  which may be inefficient and costly, as indicated by Gensim's warning about inefficiency.
