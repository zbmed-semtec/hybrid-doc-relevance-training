# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import os
import time
import logging
import utilities as utilities

def run(best_params, args, tuning=False, save_model=False):
    
    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    logging.info("Retrieved RELISH Cleaned Training Data")

    # 2) Train the model with 80% of the data (i.e. training data) and best parameters
    start = time.time()
    model = utilities.generate_Word2Vec_model(train_pmids, train_docs, best_params)
    logging.info(f"Time taken to train the model: {time.time() - start} seconds")
    logging.info("RELISH Word2Vec Model Generated.")
    logging.info("Model is being used.")

    # 3) Set the validation/test data to be used based on tuning parameter
    if tuning:
        dataset_type = "Validation"
        data_file = args.valid
        ground_truth = args.valid_ground_truth
    else:
        dataset_type = "Test"
        data_file = args.test
        ground_truth = args.test_ground_truth

    # 4) Load the data from npy file and store the tokens in a dictionary with keys as PMIDs
    article_dict = utilities.generate_npy_dict(data_file)
    logging.info(f"Prepared RELISH {data_file} Dictionary For Hybrid-Pre-WMD-Word2Vec")
     
    # 5) Generate WMD similarity: pd.DataFrame 
    similarity_df = utilities.get_WMD_similarity_scores(ground_truth, model, article_dict)
    logging.info(f"RELISH {data_file} WMD-Similarity-Matrix DataFrame Generated.")

    # 6) If the dataset type is "Test", then save the dataframes to a file each
    if dataset_type=='Test':
        similarity_file = f"output_{args.classes}/evaluation/test_similarity_{args.classes}.tsv"
        utilities.save_similarity_to_tsv(similarity_df, similarity_file)

    # 7) Save the model in the given path if specified
    if save_model:
        model_file = f"output_{args.classes}/model/Word2Vec_model_{args.classes}"
        utilities.saveWord2VecModel(model, model_file)

    return similarity_df, model