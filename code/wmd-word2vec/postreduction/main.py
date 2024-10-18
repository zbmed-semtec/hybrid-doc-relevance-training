# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py
# This file includes the modifications to the source codes according to this project!

import os
import time
import yaml
import argparse
import logging
import utilities
import precision
import calculate_gain
from train import run

#--- To remove stopwords from the MeSH-terms in MeShIDtoPMID -----
import nltk
from nltk.corpus import stopwords

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input (train) file")
    parser.add_argument("-t", "--test", help="Path to test data file")
    parser.add_argument("-v", "--valid", help="Path to validation data file")
    parser.add_argument("-gt", "--test_ground_truth", help="Path to test ground truth .tsv file")
    parser.add_argument("-gv", "--valid_ground_truth", help="Path to validation ground truth .tsv file")
    parser.add_argument("-dict", "--MeShIDtoPMID", type=str, help="Path to input MeShIDtoPMID .tsv file.")
    parser.add_argument("-c", "--classes", type=int,
                        default=3, help="Number of classes")
    parser.add_argument("-win", "--windows", type=int,
                        help="1: if using Windows systems; 0: if using Unix-like systems (including Ubuntu)")
    args = parser.parse_args()
   
    permissions = 0o755  # This sets permissions to rwxr-xr-x

    # 1) Define the directory for storing pipeline outputs
    output_directory = f"output_{args.classes}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    os.chmod(output_directory, permissions)

    # 2) Define the directory for saving the model
    model_directory = f"output_{args.classes}/model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    os.chmod(model_directory, permissions)

    # 3) Define the Directory for storing Embeddings
    embeddings_directory = f"output_{args.classes}/embeddings"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)
    os.chmod(embeddings_directory, permissions)

    # 4) Define the directory for storing validation results
    results_directory = f"output_{args.classes}/validation"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    os.chmod(results_directory, permissions)

    # 5) Define the directory for storing evaluation results
    results_directory = f"output_{args.classes}/evaluation"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    os.chmod(results_directory, permissions)

    # 6) Define the file paths to store the evaluation results
    precision_file = os.path.join(results_directory, f"precision_{args.classes}.tsv")
    dcg_file = os.path.join(results_directory, f"dcg_{args.classes}.tsv")
    idcg_file = os.path.join(results_directory, f"idcg_{args.classes}.tsv")
    ndcg_file = os.path.join(results_directory, f"ndcg_{args.classes}.tsv")

    # ---- We require this for situations where the input tokens have had stopwords removed -----------
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    #--------------------------------------------------------------------------------------------------

    # 7) Define the directory for the hyperparameter yaml file
    parameter_file = os.path.join(os.curdir, "code/wmd-word2vec/hyperparameters.yaml")
    os.chmod(parameter_file, permissions)
    with open(parameter_file, 'r') as file:
            content = yaml.safe_load(file)
            params = content['params']
            n_trials = content['iterations']['n_trials']['value']

    # 8) Run optuna optimization based on the operating system
    # Optuna can run multiple trials concurrently using n_jobs parallel processes or threads
    if args.windows:
        from optunaTuningWindows import run_optuna_optimization
        start = time.time()
        # NOTE: FOR OPTUNA HYPERPARAMETER REPRODUCIBILITY n_jobs should always be 1
        best_params, best_trial = run_optuna_optimization(args, params, n_trials, n_jobs=1)
        print("Finished optuna optimization. Time taken:", time.time()-start)
    else:
        from optunaTuningUnix import run_optuna_optimization
        start = time.time()
        # NOTE: FOR OPTUNA HYPERPARAMETER REPRODUCIBILITY n_jobs should always be 1
        best_params, best_trial = run_optuna_optimization(args, params, n_trials, n_jobs=1)
        print("Finished optuna optimization. Time taken:", time.time()-start)

    # ------------------Final Evaluation (once for test data)------------------

    # 9) Loading the model
    model_file = f"output_{args.classes}/validation/WMD_Word2Vec_best_model_{args.classes}"
    model = utilities.loadModel(model_file)

    # 10) Replacement of MeSH-terms in tokens with the corresponding MeSHIDs and store as a dictionary with keys as PMIDs
    test_article_post_annot_docs_dict = utilities.generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(args.test, args.MeShIDtoPMID)

    # 11) Generate and save the WMD similarity test matrix
    test_similarity_df = utilities.get_WMD_similarity_scores(args.test_ground_truth, model, test_article_post_annot_docs_dict)

    # 12) Save the similarity scores to a TSV file
    test_similarity_file = os.path.join(results_directory, f"test_cosine_similarity_{args.classes}.tsv") 
    utilities.save_similarity_to_tsv(test_similarity_df, test_similarity_file)

    # 13) Generate and save the precision matrix
    ref_pmids, data = precision.read_file(test_similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data, args.classes)
    precision.write_to_tsv(ref_pmids, matrix, precision_file, data)
    print("Final precision matrix saved")

    # 14) Generate and save the DCG and IDCG matrices
    sim_matrix = calculate_gain.load_cosine_sim_matrix(test_similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    print("Final DCG, IDCG, and NDCG matrices saved")