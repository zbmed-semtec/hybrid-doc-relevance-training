# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import time
from gensim.models import Word2Vec
import argparse
import logging
import utilities as utilities

def run(best_params, args, tuning=False, save_model=False):
    
    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    logging.info("Retrieved RELISH Cleaned Training Data")

    # 2) Train the model with 80% of the data (i.e. training data) and best parameters
    start = time.time()
    model = utilities.createWord2VecModel(train_pmids, train_docs, best_params)
        
    # 3) Finding MeSH-terms in training tokens to compute the corresponding MeSHIDs' embeddings and incorporate them into trained model 
    model = utilities.injection_MeSHembeddings_into_embeddings(model, train_pmids, train_docs, args.MeShIDtoPMID)
    end = time.time()
    logging.info(f"Time taken to train the model: {end - start} seconds.")
    logging.info("RELISH Word2Vec Model Generated and MeSHIDs' Embeddings Injected.")
    logging.info("Model is being used.")
        
    # 4) Set the validation/test data to be used based on tuning parameter
    if tuning:
        dataset_type = "Validation"
        data_file = args.valid
        ground_truth = args.valid_ground_truth
    else:
        dataset_type = "Test"
        data_file = args.test
        ground_truth = args.test_ground_truth

    # 5) Load the data from npy file
    pmids, docs = utilities.process_data_from_npy(data_file)
    logging.info(f"Retrieved RELISH Cleaned {dataset_type} Data")

    # 6) Replace MeSH-terms in tokens with the corresponding MeSHIDs
    docs = utilities.replacement_of_MeSHterms_with_MeSHIDs_in_tokens(pmids, docs, args.MeShIDtoPMID)   
    logging.info(f"Retrieved RELISH Cleaned {dataset_type} Data with Reduction")

    # 7) Generate the embeddings: pd.DataFrame for loaded docs
    embeddings_df = utilities.generate_document_embeddings(model, pmids, docs)
    logging.info(f"RELISH {dataset_type} Embeddings generated.") # Here similarity_file is a pd.DataFrame

    # 8) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    similarity_df = utilities.get_similarity_scores(ground_truth, embeddings_df)
    logging.info(f"RELISH {dataset_type} Cosine Similarity Matrix Generated.")

    # 9) If the dataset type is "Test", then save the dataframes to a file each
    if dataset_type=='Test':
        embeddings_file = f"output_{args.classes}/embeddings/test_embeddings_{args.classes}.pkl"
        similarity_file = f"output_{args.classes}/evaluation/test_cosine_similarity_{args.classes}.tsv"
        utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
        utilities.save_similarity_to_tsv(similarity_df, similarity_file)

    # 10) Save the model in the given path if specified
    if save_model:
        model_file = f"output_{args.classes}/model/Word2Vec_model_{args.classes}"
        utilities.saveWord2VecModel(model, model_file)

    return similarity_df, embeddings_df, model
