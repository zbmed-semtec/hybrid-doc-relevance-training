# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py
# This file includes the modifications to the source codes according to this project!

import os
import argparse
from train import run
import precision
import precision_two_class
import calculate_gain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input (train) .npy file")
    parser.add_argument("-v", "--valid", type=str, help="Path to validation data .npy file")
    parser.add_argument("-anv", "--Annot_valid", type=str, help="Path to Annotated validation data .npy file")
    parser.add_argument("-t", "--test", type=str, help="Path to test data .npy file")
    parser.add_argument("-ant", "--Annot_test", type=str, help="Path to Annotated test data .npy file")
    parser.add_argument("-gv", "--valid_ground_truth", type=str, help="Path to validation ground truth .tsv file")
    parser.add_argument("-gt", "--test_ground_truth", type=str, help="Path to test ground truth .tsv file")
    parser.add_argument("-dict", "--MeShIDtoPMID", type=str, help="Path to input MeShIDtoPMID .tsv file.")
    parser.add_argument("-rd", "--reduction", type=int,
                    help="Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0)")
    parser.add_argument("-win", "--windows", type=int,
                    help="1: if using Windows systems && 0: if using Unix-like systems (including Ubuntu)")
    args = parser.parse_args()
    
    # Define the directory for storing pipeline outputs
    output_directory = "output_of_model"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    # Define the directory for saving the model
    model_directory = "output_of_model/model"
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    # Define the Directory for Storing Embeddings
    embeddings_directory = "output_of_model/doc_embeddings"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)
    # Define the directory for storing evaluation results
    results_directory = "output_of_model/evaluation"
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    if args.windows:
        from optunaTuning_Windows import run_optuna_optimization
        best_params, best_trial = run_optuna_optimization(args, n_trials=100, n_jobs=2)
    else:
        from optunaTuning_Unix import run_optuna_optimization
        best_params, best_trial = run_optuna_optimization(args, n_trials=100, n_jobs=2)
        
    """
    #----- Manually best_params given -------
    best_params = {
        "vector_size": 360,
        "window": 12,
        "min_count": 2,
        "epochs": 15,
        "workers": 2,
        "sg" : 1
        }
    """
    print("Finished Optuna optimization and Start Evaluation Test-data and Saving the Best Model")
    
    similarity_file, embeddings_file, model = run(best_params, args, tuning=False, save_model=True)
    
    # In case of using the same data for test and tunning one, instead of the line above, can use the line below
    #similarity_file = "output_of_model/evaluation/best_cosine_similarity.tsv"
    
    # Define the file paths for the pipeline evaluation results
    precision_file = os.path.join(results_directory, "precision_three_class.tsv")
    precision_file_two_class = os.path.join(results_directory, "precision_two_class.tsv")
    dcg_file = os.path.join(results_directory, "dcg.tsv")
    idcg_file = os.path.join(results_directory, "idcg.tsv")
    ndcg_file = os.path.join(results_directory, "ndcg.tsv")

    # Generate and save the three-class precision matrix
    ref_pmids, data = precision.read_file(similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data)
    precision.write_to_tsv(ref_pmids, matrix, precision_file)
    print("Three-class Precision matrix saved")
    
    # Generate and save the two-class precision matrix
    ref_pmids, data = precision_two_class.read_file(similarity_file)
    matrix = precision_two_class.generate_matrix(ref_pmids, data)
    precision_two_class.write_to_tsv(ref_pmids, matrix, precision_file_two_class)
    print("Two-class Precision matrix saved")

    # Generate and save the DCG and IDCG matrices
    sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    print("DCG, IDCG, and NDCG matrices saved")



