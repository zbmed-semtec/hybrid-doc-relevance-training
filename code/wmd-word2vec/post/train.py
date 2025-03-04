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

    # 2) Train the model with 80% of the data (i.e. training data) and best parameters
    start = time.time()
    model = utilities.createWord2VecModel(train_pmids, train_docs, best_params)

    # 3) Finding MeSH-terms in training tokens to compute the corresponding MeSHIDs' embeddings and incorporate them into trained model
    model = utilities.injection_MeSHembeddings_into_embeddings(model, train_pmids, train_docs, args.MeShIDtoPMID)
    end = time.time()
    logging.info(f"Time taken to train the model: {end - start} seconds.")
    logging.info("RELISH Word2Vec Model Generated and MeSHIDs' Embeddings Injected.")
    logging.info("Model is being used.")

    # 4) Store Relish Post-processed/annotated tokens in a dictionary with keys PMIDs
    val_article_post_annot_docs_dict = utilities.generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(args.valid, args.MeShIDtoPMID)
    logging.info(f"Prepared RELISH Post-Annotation Validation Dictionary For Hybrid-WMD-Post-Word2Doc2Vec")
        
    # 5) Generate WMD similarity: pd.DataFrame 
    val_similarity_df = utilities.get_WMD_similarity_scores(args.valid_ground_truth, model, val_article_post_annot_docs_dict)
    logging.info(f"RELISH Validation WMD-Similarity-Matrix DataFrame Generated.")
    
    return val_similarity_df, model