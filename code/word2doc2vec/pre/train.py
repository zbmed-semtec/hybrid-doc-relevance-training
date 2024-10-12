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
    logging.info("RELISH Hybrid Word2Vec Model Generated.")
    logging.info("Model is being used.")

    # 3) Load the data from npy file
    val_pmids, val_docs = utilities.process_data_from_npy(args.valid)
    logging.info(f"Retrieved RELISH Cleaned Validation Data")
        
    # 4) Generate the embeddings: pd.DataFrame for loaded docs
    val_embeddings_df = utilities.generate_document_embeddings(model, val_pmids, val_docs)
    logging.info(f"RELISH Validation Embeddings Pickle File Generated.")

    # 5) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    val_similarity_df = utilities.get_similarity_scores(args.valid_ground_truth, val_embeddings_df)
    logging.info(f"RELISH Validation Cosine Similarity Matrix Generated.")

    return val_similarity_df, val_embeddings_df, model