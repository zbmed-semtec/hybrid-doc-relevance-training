# Source code: 
# https://github.com/zbmed-semtec/doc2vec-doc-relevance-training/blob/main/code/train_model/train.py
# This file includes the modifications to the source codes according to this project!

import time
import logging
import utilities as utilities

def run(best_params, args, save_model=False):
    
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

    # 4) Set the test data to be used based on tuning parameter
    dataset_type = "Test"
    data_file = args.test
    ground_truth = args.ground_truth

    # 5) Store Relish Post-processed/annotated tokens in a dictionary with keys PMIDs
    article_post_annot_docs_dict = utilities.generate_post_npy_dict_via_injection_MeSHIDs_into_tokens(data_file, args.MeShIDtoPMID)
    logging.info(f"Prepared RELISH Post-Annotation {data_file} Dictionary For Hybrid-WMD-Post-Word2Doc2Vec")
        
    # 6) Generate WMD similarity: pd.DataFrame 
    similarity_df = utilities.get_WMD_similarity_scores(ground_truth, model, article_post_annot_docs_dict)
    logging.info(f"RELISH {data_file} WMD-Similarity-Matrix DataFrame Generated.")

    # 7) If the dataset type is "Test", then save the dataframes to a file each
    if dataset_type== 'Test':
        similarity_file = f"output_{args.classes}/evaluation/test_wmd_similarity_{args.classes}.tsv"
        utilities.save_similarity_to_tsv(similarity_df, similarity_file)

    # 8) Save the model in the given path if specified
    if save_model:
        model_file = f"output_{args.classes}/model/WMD_Word2Vec_model_{args.classes}"
        utilities.saveWord2VecModel(model, model_file)
    
    return similarity_df, model