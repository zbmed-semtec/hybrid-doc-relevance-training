**Note that in this directory, models are trained and embeddings are generated for data with Removed-Stopwords**

In order to generalize the code for data including Stopwords, some changes are required in script [utilities.py](./code/utilities.py). Specifically, `pattern_to_find = [w for w in pmid_term[1:] if not w in stop_words]` in line 151 of function `generate_postReduction_npy_dict_via_injection_MeSHIDs_into_tokens`, line 215 of function `generate_post_npy_dict_via_injection_MeSHIDs_into_tokens`, line 373 of function `replacement_of_MeSHterms_with_MeSHIDs_in_tokens`, and line 417 of function `injection_MeSHIDs_into_tokens` must be replaced with `pattern_to_find = pmid_term[1:]`.

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

#### Navigate to the directory `hybrid-post-word2doc2vec-doc-relevance`:

```
cd hybrid-post-word2doc2vec-doc-relevance-training/hybrid-post-word2doc2vec-doc-relevance
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
- Download the dataset from this link: [RELISH_Tokenized_Removed_Stopwords](https://drive.google.com/file/d/1TrnWdIWlrLTCEXBeV3x8rFwd4wKVdmy9/view)

### Step 4: Create Model Pipeline

This pipeline aims to create Word2Vec models using given hyperparameter sets, train the models, and evaluate their performance using three-class precision, two-class precision and nDCG scores.

#### Pipeline Steps:

- **Model Training**: Trains Word2Vec model with the given hyperparameters using the input tokens.
- **WMD Similarity Computation**: Calculates [WMD (Word Mover’s Distance)](../docs/WMD.md) similarities for input dataset using Gensim implementation of WMD, which utilizes the trained Word2Vec model.
- **Embedding Generation**: Generates embeddings for input documents using the trained model.
- **Cosine Similarity Computation**: Calculates cosine similarities for the generated embeddings.
- **Precision@N Calculation**: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained WMD similarities and for the obtained cosine similarities.
- **NDCG Score Calculation**: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this [script](./code/main.py), and run the following command:

```
python3 code/main.py [-i INPUT_DATA] [-gt GROUND_TRUTH] [-pj PARAMS_JSON] [-dict MeShIDtoPMID] [-rd REDUCTION_OR_NOT] [-cos COMPUTING_COSINE_SIMILARITY] [-wmd COMPUTING_WMD_SCORE]
```

You must pass the following four arguments:

+ -i/ --input : File path to input dataset (.npy file format).
+ -gt/ --ground_truth : File path for the ground truth (.tsv file format).
+ -pj/ --params_json: File path to word2vec hyperparameters JSON.
+ -dict/ --MeShIDtoPMID : File path to input MeShIDtoPMID (.tsv file format).
+ -rd/ --reduction : Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0).
+ -cos/ --cosine_similarity : Whether to calculate the documents' cosine similarity (1) or not (0).
+ -wmd/ --WMD_score : Whether to calculate the documents' WMD score (1) or not (0).

For instance, to run this script for _hybrid-post-Word2Doc2Vec_ and _hybrid-post-WMD-Word2Vec_, you may execute the following command:

```
python3 code/main.py -i RELISH_Tokenized_Removed_Stopwords.npy -gt ../data/RELISH_ground_truth.tsv -pj ../data/hyperparameters_word2vec.json -dict ../data/dic_MeShIDtoPMID_2022628.tsv -rd 0 -cos 1 -wmd 1
```

In case of hybrid-PostReduction-Word2Doc2Vec_ and _hybrid-PostReduction-WMD-Word2Vec_, you may execute the following command:
```
python3 code/main.py -i RELISH_Tokenized_Removed_Stopwords.npy -gt ../data/RELISH_ground_truth.tsv -pj ../data/hyperparameters_word2vec.json -dict ../data/dic_MeShIDtoPMID_2022628.tsv -rd 1 -cos 1 -wmd 1
```

All outputs of the [script](./code/main.py) are saved in the folder named `output_of_model`. For hyperparameter set i, the code saves the corresponding trained model, embeddings, cosine similarities, and WMD scores in the following file paths: `output_of_model/model_i/Word2Vec_model`, `output_of_model/doc_embeddings/embeddings_pickle_i.pkl`, `output_of_model/evaluation/cosine_similarity_i.tsv`, and `output_of_model/evaluation/WMD_scores/WMD_similarity_i.tsv`, respectively. Additionally, evaluation results using cosine similarities and WMD scores are stored in the folders `output_of_model/evaluation` and `output_of_model/evaluation/WMD_scores`, correspondingly, with the same naming convention.

#### Step 5: Compile Results

In order to compile the average result of each hyperparameter set for Precison@ and nDCG@N and generate a single TSV file each, please use this [script](code/evaluation/show_avg.py).

You must pass the following two arguments:

+ -i / --input: Path to the directory consisting of all the precision matrices/gain matrices.
+ -o/ --output: Output path along with the name of the file to save the generated compiled Precision@N / nDCG@N TSV file.

Make sure to move all precision (class distribution wise) and gain files to separate folders before executing this script.

If you are running the code from the code folder, run the compilation script as:

```
python3 code/evaluation/show_avg.py -i data/output/gain_matrices/ -o data/output/results_gain.tsv
```

NOTE: Please do not forget to put a `'/'` at the end of the input file path. Execute the above script for both gain and precision results.
