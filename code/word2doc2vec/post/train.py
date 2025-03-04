# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import time
import logging
import utilities as utilities

def run(best_params, args):
    
    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    logging.info("Retrieved RELISH Cleaned Training Data")

    # 2) Train the model with 90% of the data (i.e. training data) and best parameters
    start = time.time()
    model = utilities.createWord2VecModel(train_pmids, train_docs, best_params)

    # 3) Finding MeSH-terms in training tokens to compute the corresponding MeSHIDs' embeddings and incorporate them into trained model
    model = utilities.injection_MeSHembeddings_into_embeddings(model, train_pmids, train_docs, args.MeShIDtoPMID)
    end = time.time()
    logging.info(f"Time taken to train the model: {end - start} seconds.")
    logging.info("RELISH Word2Vec Model Generated and MeSHIDs' Embeddings Injected.")
    logging.info("Model is being used.")
        
    # 4) Load the validation data from npy file
    val_pmids, val_docs = utilities.process_data_from_npy(args.valid)
    logging.info(f"Retrieved RELISH Cleaned Validation Data")

    # 5) Find MeSH-terms in the data in order to append the corresponding MeSHIDs' to tokens
    val_docs = utilities.injection_MeSHIDs_into_tokens(val_pmids, val_docs, args.MeShIDtoPMID)
    logging.info("Injected MeSH IDs as tokens")

    # 6) Generate the embeddings for validation dataset: pd.DataFrame for loaded docs
    val_embeddings_df = utilities.generate_document_embeddings(model, val_pmids, val_docs)
    logging.info(f"RELISH Validation Embeddings generated.") # Here similarity_file is a pd.DataFrame

    # 7) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    val_similarity_df = utilities.get_similarity_scores(args.valid_ground_truth, val_embeddings_df)
    logging.info(f"RELISH Validation Cosine Similarity Matrix Generated.")

    return val_similarity_df, val_embeddings_df, model
