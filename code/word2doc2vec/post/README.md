# Hybrid-Post-Word2doc2vec

This directory contains the code for the Hybrid-Post-Word2doc2vec approach. This method leverages the plain text of the RELISH corpus, utilizes the word2vec model to generate word embeddings, and calculates their centroid to produce document embeddings.

# Input Data
The input data for this approach includes plain text (title and abstract) of the documents within the RELISH Corpus. The plain text is then converted into preprocessed tokens and are stored in the RELISH.npy file, which contains preprocessed arrays of PMIDs, document titles, and abstracts. These arrays are produced through an extensive preprocessing pipeline, detailed in the (relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#text-preprocessing-for-generating-embeddings). The preprocessed tokenized data is split into training, validation and test datasets based on specific criteria used in the splitting algorithm as explained [here](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#splitting-the-data).


## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository
First, clone the repository to your local machine using one of the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/hybrid-doc-relevance-training.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/hybrid-doc-relevance-training.git
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
The code is stable with python 3.9 and higher. The required python packages are listed in the requirements.txt file. To install the required packages, run the following command:

```
pip install -r requirements.txt
```

To deactivate the virtual environment after running the project, run the following command:

```
deactivate
```

### Step 3: Dataset


- Use the [Download_Dataset.sh](./Download_Dataset.sh) script to download the Split Dataset by running the following commands:

```
chmod +777 Download_Dataset.sh
./Download_Dataset.sh
```
This script makes sure that the necessary folders are created and the files are downloaded in the corresponding folders.

**OR**


- You could also download the dataset from this link: [Split_Dataset](https://drive.google.com/drive/folders/1Bq_U5207utn7tvSt_HLVdOdYR5QW7MMN). Please make sure to keep the data in the below specified format.

```
📦 /hybrid-doc-relevance-training
└─ data
   └─ Split_Dataset
      ├─ Data
      │  ├─ train.npy
      │  ├─ test.npy
      │  ├─ valid.npy
      └─ Ground_truth
         ├─ train.tsv
         ├─ test.tsv
         └─ valid.tsv
```

### Step 4: Optimization Pipeline

This step optimizes hyperparameters for a Word2vec model using Optuna, train the model with the optimal parameters, and evaluate its performance using precision at N (Precision@N) and normalized discounted cumulative gain (NDCG) metrics.

Pipeline Steps:
+ Hyperparameter Optimization: Utilizes Optuna to search for the best hyperparameters for the Word2vec model.
+ Model Training: Trains the Word2vec model with the optimal hyperparameters using 90% training split data.
+ Embedding Generation: Generates embeddings for 5% validation split data using the trained model.
+ Cosine Similarity Computation: Calculates cosine similarities for the generated embeddings.
+ Precision@N Calculation: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
+ NDCG Score Calculation: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this script, and run the following command:

 ``` 
python3 code/word2doc2vec/post/main.py [-i INPUT] [-t TEST_FILE] [-v VALIDATION_FILE] [-gt TEST_GROUND_TRUTH_FILE] [-gv VALIDATION_GROUND_TRUTH_FILE] [-dict MeShIDtoPMID]  [-c NO_OF CLASSES] [-win WINDOWS/LINUX]
 ``` 

 You must pass the following four arguments:

+ -i/ --input : File path to the RELISH Train split dataset (.npy file format).
+ -t/ --test :  File path to the RELISH Test split dataset (.npy file format).
+ -v/ --valid: File path to the RELISH Validation split dataset (.npy file format).
+ -gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
+ -gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
+ -dict/ --MeShIDtoPMID : File path to input MeShIDtoPMID (.tsv file format).
+ -c/  --classes : No. of classes to perform optimization on (Integer 2 or 3/ Default value is 3)
+ -win/ --windows : 1- if using Windows systems; 0- if using Unix-like systems (including Ubuntu)

To run this script, please execute the following command:

```
python3 code/word2doc2vec/post/main.py -i data/Split_Dataset/Data/train.npy -t data/Split_Dataset/Data/test.npy -v data/Split_Dataset/Data/valid.npy -gt data/Split_Dataset/Ground_truth/test.tsv -gv data/Split_Dataset/Ground_truth/valid.tsv -dict data/mesh_to_pmid_dict.tsv -c 3 -win 0 
```

### NOTE:

Unfortunately, using `model.wv.add_vectors` to add new MeSHembedding vectors in batches in the `injection_MeSHembeddings_into_embeddings` function of the [utilities.py](./code/utilities.py) script causes the best model stored from the tuning phase to be unloaded. Also, using `model.wv.add_vector()` to add single vectors to KeyedVectors, which grows by one each time, significantly reduces the execution speed of the code, as indicated by Gensim's inefficiency warning.

Therefore to add new MeSHembedding-vectors, preallocating space for the required size is utilized, which allocates memory for the entire set of vectors upfront, instead of dynamically resizing the storage as vectors are added. This approach can help mitigate the inefficiency associated with adding vectors one by one. **However vocabulary attributes may not be correspondingly updated!**
