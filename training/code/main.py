# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py
# This file includes the modifications to the source codes according to this project!

import os
import argparse
from optunaTuning import run_optuna_optimization
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
    args = parser.parse_args()
    
    # Define the directory for storing model results
    output_directory = "output_of_model"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    best_params, best_trial = run_optuna_optimization(args, n_trials=100, n_jobs=2)

    print("Finished Optuna optimization and Start Evaluation Test-data and Saving the Best Model")
    similarity_file = run(best_params, args, tuning=False, save_model=True)

    #output_directory = "output_of_model/evaluation"
    output_directory = os.path.join(output_directory, "evaluation")
    
    precision_file = os.path.join(output_directory, "precision_three_class.tsv")
    precision_file_two_class = os.path.join(output_directory, "precision_two_class.tsv")
    dcg_file = os.path.join(output_directory, "dcg.tsv")
    idcg_file = os.path.join(output_directory, "idcg.tsv")
    ndcg_file = os.path.join(output_directory, "ndcg.tsv")

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



