# Hybrid-Pre-Word2doc2vec
This directory contains the code for the Hybrid-Pre-Word2doc2vec approach. This method leverages the pre-annotated text of the RELISH corpus, utilizes the word2vec model to generate word embeddings, and calculates their centroid to produce document embeddings.

## Input Data

The input data for this approach includes text annotated using the MeSH vocabulary via the [Whatizit tool](https://academic.oup.com/bioinformatics/article/24/2/296/227269?login=true). The complete annotation pipeline is detailed in the [repository documentation](https://github.com/zbmed-semtec/whatizit-dictionary-ner/tree/main/docs). The annotated text is then converted into preprocessed tokens and are stored in the RELISH.npy file, which contains preprocessed arrays of PMIDs, document titles, and abstracts. These arrays are produced through an extensive preprocessing pipeline, detailed in the [relish-preprocessing repository](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#text-preprocessing-for-generating-embeddings). The preprocessed tokenized data is split into training, testing, and validation datasets based on specific criteria used in  the splitting algorithm as explained [here](https://github.com/zbmed-semtec/relish-preprocessing?tab=readme-ov-file#splitting-the-data). For a detailed explanation of the data annotation, data preprocessing and the algorithm please refer [here](../../../README.md).

## Getting Started

To get started with this project, follow these steps:

### Step 1: Clone the Repository

First, clone the repository to your local machine using the following command:

###### Using HTTP:

```
git clone https://github.com/zbmed-semtec/hybrid-doc-relevance-training.git
```

###### Using SSH:
Ensure you have set up SSH keys in your GitHub account.

```
git clone git@github.com:zbmed-semtec/hybrid-doc-relevance-training.git
```

### Step 2: Create a Virtual Environment and Install Dependencies

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


- Use the [Download_Data.sh](./Download_Data.sh) script to download the Split Dataset by running the following commands:

```
chmod +777 Download_Data.sh
./Download_Data.sh
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
      │  └─ valid.npy
      └─ Ground_truth
         ├─ train.tsv
         ├─ test.tsv
         └─ valid.tsv
```


### Step 4: Generate Embeddings

This step optimizes hyperparameters for a Word2vec model using Optuna, train the model with the optimal parameters, and evaluate its performance using precision at N (Precision@N) and normalized discounted cumulative gain (NDCG) metrics.

Pipeline Steps:
+ Hyperparameter Optimization: Utilizes Optuna to search for the best hyperparameters for the Word2vec model.
+ Model Training: Trains the Word2vec model with the optimal hyperparameters using 80% of the training split data.
+ Embedding Generation: Generates embeddings for the remaining 20% of the test split data using the trained model.
+ Cosine Similarity Computation: Calculates cosine similarities for the generated embeddings.
+ Precision@N Calculation: Computes Precision@N scores, a measure of the relevance of retrieved documents, for the obtained cosine similarities.
+ NDCG Score Calculation: Computes normalized discounted cumulative gain (NDCG) scores, which assesses the quality of ranked search results based on relevance assessments.

In order to start the pipeline execution use this script, and run the following command:

 ``` 
python3 code/pre/word2doc2vec/main.py [-i INPUT] [-v VALIDATION_FILE] [-t TEST_FILE] [-gv VALIDATION_GROUND_TRUTH] [-gt TEST_GROUND_TRUTH] [-c NO_OF CLASSES] [-win WINDOWS/LINUX]
 ``` 

 You must pass the following four arguments:

+ -i/ --input : File path to the RELISH Train split dataset (.npy file format).
+ -v/ --valid : File path to the RELISH Validation split dataset (.npy file format).
+ -t/ --test : File path to the RELISH Test split dataset (.npy file format).
+ -gv/ --valid_ground_truth : File path for the Validation split ground truth (.tsv file format).
+ -gt/ --test_ground_truth : File path for the Test split ground truth (.tsv file format).
+ -c/ --classes : No. of classes to perform optimization on (Integer 2 or 3/ Default value is 3).
+ -win/ --windows : 1 - if using Windows systems; 0 - if using Unix-like systems (including Ubuntu)

To run this script, please execute the following command:

 ``` 
python3 code/pre/word2doc2vec/main.py -i data/Split_Dataset/train.npy -v data/Split_Dataset/valid.npy -t data/Split_Dataset/test.npy -gv data/Split_Dataset/Ground_truth/valid.tsv -gt data/Split_Dataset/Ground_truth/test.tsv -c 3 -win 0
 ``` 

Precision@N and NDCG scores are saved as TSV files in the following folder path: `\output_2\evaluation\`  for 2 class distribution and `\output_3\evaulation\` for 3 class distribution for further analysis and reporting.

Make sure to run the model training twice for both the class distributions by changing the value of the -c/ --classes flag to 2 and 3.

**NOTE:** As of now, we use the test file as our validation dataset during the model training. Make sure to replace the validation dataset with the truth dataset as well as validation groundtruth file with the test groundtruth file.

For replacing the validation data with the test data, please execute the following command:

``` 
python3 code/pre/word2doc2vec/main.py -i data/Split_Dataset/train.npy -v data/Split_Dataset/test.npy -t data/Split_Dataset/test.npy -gv data/Split_Dataset/Ground_truth/test.tsv -gt data/Split_Dataset/Ground_truth/test.tsv -c 3 -win 0
``` 