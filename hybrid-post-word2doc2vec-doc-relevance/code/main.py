# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/main.py
# This file includes the modifications to the source codes according to this project!

import os
import argparse
import json
import time
from train import run
import precision
import precision_two_class
import calculate_gain

#--- To remove stopwords from the MeSH-terms in MeShIDtoPMID -----
import nltk
from nltk.corpus import stopwords

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, help="Path to input dataset .npy file")
    parser.add_argument("-gt", "--ground_truth", type=str, help="Path to ground truth .tsv file")
    parser.add_argument("-pj", "--params_json", type=str, help="File path to word2vec hyperparameters JSON")
    parser.add_argument("-dict", "--MeShIDtoPMID", type=str, help="Path to input MeShIDtoPMID .tsv file.")
    parser.add_argument("-rd", "--reduction", type=int,
                    help="Whether to reduce the documents'words by replacing the catalogued ones with corresponding MeSHID (1) or not (0)")
    parser.add_argument("-cos", "--cosine_similarity", type=int,
                    help="Whether to calculate the documents' cosine similarity (1) or not (0)")
    parser.add_argument("-wmd", "--WMD_score", type=int,
                    help="Whether to calculate the documents' WMD score (1) or not (0)")
    
    args = parser.parse_args()
    
    # Extract model hyperparameters
    params = []
    with open(args.params_json, "r") as openfile:
        params = json.load(openfile)
    
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
    # Define the directory for storing evaluation results, specifically cosine-similarity results
    results_directory = os.path.join(output_directory, "evaluation")
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)
    # Define the directory for storing evaluation results of WMD-similarity
    WMD_results_directory = os.path.join(results_directory, "WMD_scores")
    if not os.path.exists(WMD_results_directory):
        os.makedirs(WMD_results_directory)

    # ---- We require this for situations where the input tokens have had stopwords removed -----------
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    #--------------------------------------------------------------------------------------------------
    
    for iteration in range(0, len(params)):
        print(
            f'start for parameter-set {iteration} with reduction={args.reduction}, using WMD={args.WMD_score}, and cosine-similarity={args.cosine_similarity}'
        )
              
        start = time.time()
        os.makedirs(f"output_of_model/model/{iteration}", exist_ok=True)
              
        run(params[iteration], args, iteration, save_model=True)
              
        if args.cosine_similarity:
            similarity_file = f"output_of_model/evaluation/cosine_similarity_{iteration}.tsv"
    
            precision_file = os.path.join(results_directory, f"precision_three_class_{iteration}.tsv")
            precision_file_two_class = os.path.join(results_directory, f"precision_two_class_{iteration}.tsv")
            dcg_file = os.path.join(results_directory, f"dcg_{iteration}.tsv")
            idcg_file = os.path.join(results_directory, f"idcg_{iteration}.tsv")
            ndcg_file = os.path.join(results_directory, f"ndcg_{iteration}.tsv")

            # Generate and save the three-class precision matrix
            ref_pmids, data = precision.read_file(similarity_file)
            matrix = precision.generate_matrix(ref_pmids, data)
            precision.write_to_tsv(ref_pmids, matrix, precision_file)
            print(f"Three-class Precision matrix using cos-sim saved for parameter-set {iteration}")
    
            # Generate and save the two-class precision matrix
            ref_pmids, data = precision_two_class.read_file(similarity_file)
            matrix = precision_two_class.generate_matrix(ref_pmids, data)
            precision_two_class.write_to_tsv(ref_pmids, matrix, precision_file_two_class)
            print(f"Two-class Precision matrix using cos-sim saved for parameter-set {iteration}")

            # Generate and save the DCG and IDCG matrices
            sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
            calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
            calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
            all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
            calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
        
            print(f"DCG, IDCG, and NDCG matrices using cos-sim saved for parameter-set {iteration}")
        
            # Delete dcg and idcg files whose results are summarized in ndcg
            try:
                os.remove(os.path.join(results_directory, f"dcg_{iteration}.tsv"))
                os.remove(os.path.join(results_directory, f"idcg_{iteration}.tsv"))
            except FileNotFoundError:
                continue
              
        if args.WMD_score:
            similarity_file = f"output_of_model/evaluation/WMD_scores/WMD_similarity_{iteration}.tsv"
    
            precision_file = os.path.join(WMD_results_directory, f"precision_three_class_{iteration}.tsv")
            precision_file_two_class = os.path.join(WMD_results_directory, f"precision_two_class_{iteration}.tsv")
            dcg_file = os.path.join(WMD_results_directory, f"dcg_{iteration}.tsv")
            idcg_file = os.path.join(WMD_results_directory, f"idcg_{iteration}.tsv")
            ndcg_file = os.path.join(WMD_results_directory, f"ndcg_{iteration}.tsv")

            # Generate and save the three-class precision matrix
            ref_pmids, data = precision.read_file(similarity_file)
            matrix = precision.generate_matrix(ref_pmids, data)
            precision.write_to_tsv(ref_pmids, matrix, precision_file)
            print(f"Three-class Precision matrix using WMD-score saved for parameter-set {iteration}")
    
            # Generate and save the two-class precision matrix
            ref_pmids, data = precision_two_class.read_file(similarity_file)
            matrix = precision_two_class.generate_matrix(ref_pmids, data)
            precision_two_class.write_to_tsv(ref_pmids, matrix, precision_file_two_class)
            print(f"Two-class Precision matrix using WMD-score saved for parameter-set {iteration}")

            # Generate and save the DCG and IDCG matrices
            sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
            calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
            calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
            all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
            calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
        
            print(f"DCG, IDCG, and NDCG matrices using WMD-score saved for parameter-set {iteration}")
        
            # Delete dcg and idcg files whose results are summarized in ndcg
            try:
                os.remove(os.path.join(WMD_results_directory, f"dcg_{iteration}.tsv"))
                os.remove(os.path.join(WMD_results_directory, f"idcg_{iteration}.tsv"))
            except FileNotFoundError:
                continue
              
        end = time.time()
        print(f"Time Taken for parameter-set {iteration}: {end - start} seconds.")
