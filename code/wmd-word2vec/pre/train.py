# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import time
import logging
import utilities as utilities

def run(best_params, args):
    
    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    logging.info("Retrieved RELISH Cleaned Training Data")

    # 2) Train the model with 80% of the data (i.e. training data) and best parameters
    start = time.time()
    model = utilities.generate_Word2Vec_model(train_pmids, train_docs, best_params)
    logging.info(f"Time taken to train the model: {time.time() - start} seconds")
    logging.info("RELISH Word2Vec Model Generated.")
    logging.info("Model is being used.")

    # 3) Load the data from npy file and store the tokens in a dictionary with keys as PMIDs
    val_article_dict = utilities.generate_npy_dict(args.valid)
    logging.info(f"Prepared RELISH Validation Dictionary For Hybrid-Pre-WMD-Word2Vec")
     
    # 4) Generate WMD similarity: pd.DataFrame 
    val_similarity_df = utilities.get_WMD_similarity_scores(args.valid_ground_truth, model, val_article_dict)
    logging.info(f"RELISH Validation WMD-Similarity-Matrix DataFrame Generated.")

    return val_similarity_df, model