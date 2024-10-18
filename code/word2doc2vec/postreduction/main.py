# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py
# This file includes the modifications to the source codes according to this project!

import os
import yaml
import time
import argparse
import precision
import logging
import utilities
import calculate_gain

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
    parameter_file = os.path.join(os.curdir, "code/word2doc2vec/hyperparameters.yaml")
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
    model_file = f"output_{args.classes}/validation/Word2Vec_best_model_{args.classes}"
    model = utilities.loadModel(model_file)

    # 10) Loading test data
    test_pmids, test_docs = utilities.process_data_from_npy(args.test)

    # 11) Replace MeSH-terms in tokens with the corresponding MeSHIDs
    test_docs = utilities.replacement_of_MeSHterms_with_MeSHIDs_in_tokens(test_pmids, test_docs, args.MeShIDtoPMID)   

    # 12) Generate the embeddings: pd.DataFrame for loaded docs
    test_embeddings_df = utilities.generate_document_embeddings(model, test_pmids, test_docs)
    
    # 13) Save the embeddings to a pickle file
    test_embedding_file = os.path.join(embeddings_directory, f"test_embeddings_{args.classes}.pkl")
    utilities.save_embeddings_to_pickle(test_embeddings_df, test_embedding_file)

    # 14) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    test_similarity_df = utilities.get_similarity_scores(args.test_ground_truth, test_embeddings_df)

    # 15) Save the similarity scores to a TSV file
    test_similarity_file = os.path.join(results_directory, f"test_cosine_similarity_{args.classes}.tsv") 
    utilities.save_similarity_to_tsv(test_similarity_df, test_similarity_file)

    # 16) Generate and save the precision matrix
    ref_pmids, data = precision.read_file(test_similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data, args.classes)
    precision.write_to_tsv(ref_pmids, matrix, precision_file, data)
    print("Final precision matrix saved")

    # 17) Generate and save the DCG and IDCG matrices
    sim_matrix = calculate_gain.load_cosine_sim_matrix(test_similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    print("Final DCG, IDCG, and NDCG matrices saved")