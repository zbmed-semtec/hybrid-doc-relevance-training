import os
'''
By properly implementing file locking mechanisms like using fcntl for Unix-like systems ( and msvcrt for Windows systems), one can ensure that the Optuna optimization process runs smoothly without encountering race conditions or file locking issues, even when using multiple processes (n_jobs > 1).
'''
#import msvcrt # For Windows systems
import fcntl # For Unix-like systems (including Ubuntu)

import threading
# Define a lock for synchronization
precision_lock = threading.Lock()

import optuna
import argparse
import logging
import pandas as pd
import numpy as np
from train import run
import utilities as utilities
from tqdm import tqdm
import precision as precision
import pickle

# Define the directory for storing log-files
log_directory = "output_of_model"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

log_file = "output_of_model/Optuna_trials.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# To save the best validation's trained model and its corresponding embeddings and similarities
def save_best_model_and_embeddings_and_similarity(model, embeddings_df, similarity_df):
    # Define the file path for saving the best validation model
    model_file = "output_of_model/model/best_Word2Vec_model"
    # Define the file path for Storing Embeddings of the Best Validation Hyperparameter Set
    embeddings_file_dest = "output_of_model/doc_embeddings/best_embeddings_pickle.pkl"
    # Define the file path for the best similarity file
    similarity_file_dest = "output_of_model/evaluation/best_cosine_similarity.tsv"
    
    # Acquire a file lock before accessing or modifying the model file
    with open(model_file, "w") as model_lock_file:
        fcntl.flock(model_lock_file.fileno(), fcntl.LOCK_EX)
        # Save the model
        utilities.saveWord2VecModel(model, model_file)
        # Release the file lock when done
        fcntl.flock(model_lock_file.fileno(), fcntl.LOCK_UN)

    # Acquire a file lock before accessing or modifying the embeddings file
    with open(embeddings_file_dest, "w") as embeddings_lock_file:
        fcntl.flock(embeddings_lock_file.fileno(), fcntl.LOCK_EX)
        # Save Embeddings for the Best Validation Hyperparameter Set
        utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file_dest)
        # Release the file lock when done
        fcntl.flock(embeddings_lock_file.fileno(), fcntl.LOCK_UN)

    # Acquire a file lock before accessing or modifying the similarity file
    with open(similarity_file_dest, "w") as similarity_lock_file:
        fcntl.flock(similarity_lock_file.fileno(), fcntl.LOCK_EX)
        # Save the best similarity file as a csv file
        utilities.save_similarity_scores(similarity_df, similarity_file_dest)
        # Release the file lock when done
        fcntl.flock(similarity_lock_file.fileno(), fcntl.LOCK_UN)

    
    
# Define Objective Function for Optimization
def objective_wrapper(args):
    def objective(trial):
        # Suggest hyperparameters for Word2Vec
        vector_size = trial.suggest_int('vector_size', 100, 500, step=50)
        window = trial.suggest_int('window', 5, 15)
        min_count = trial.suggest_int('min_count', 1, 6)
        epochs = trial.suggest_int('epochs', 5, 15)
        workers = 8 # trial.suggest_int('workers', 2, 8)
        sg = trial.suggest_int('sg', 0, 1)

        # Use args here as needed, e.g., args.input, args.test
        params = {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "workers": workers,
            "sg" : sg
        }

        # Assume run() trains the model and returns the paths to files with similarity scores,embeddings and the trained model
        similarity_df, embeddings_df, model = run(params, args, tuning = True, save_model=False)
        
        #ref_pmids, data = precision.read_file(similarity_file)
        ref_pmids = similarity_df["PID1"].unique()
        
        vector = precision.generate_vector(ref_pmids, similarity_df)

        precision_5 = list(np.mean(vector, axis=0).round(4))
        
        # Load the previously saved best precision value
        best_precision_path = "output_of_model/best_precision.txt"
        if os.path.exists(best_precision_path):
            with open(best_precision_path, "r") as f:
                best_precision = float(f.read().strip()) # .strip() removes leading and trailing whitespace characters from a string.
        else:
            best_precision = -1.0

        
        """
        When trial.should_prune() returns True, it means that Optuna has determined that the current trial is not likely to produce 
        a better result compared to the previous trials. As a result, it suggests pruning (stopping) this trial early to save 
        computational resources.
        """
        # To avoid unnecessary computations and file-saving operations for trials that are suggested for pruning
        if trial.should_prune(): #should_prune() does not support multi-objective optimization: it only considers a single objective/metric
            return precision_5
        
        # Acquire the lock before updating best_precision
        with precision_lock:
            if precision_5[0] > best_precision:
                best_precision = precision_5[0]  # Update the best precision

                # Save the best model and its corresponding embeddings and similarity files
                save_best_model_and_embeddings_and_similarity(model, embeddings_df, similarity_df)
                print('SAVINGGGGGGGG')

                # Save the new best precision value
                with open(best_precision_path, "w") as f:
                    f.write(str(best_precision))
        
        return precision_5
    return objective

def run_optuna_optimization_without_saving_study(args, n_trials=10, n_jobs=1):
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials) as pbar:
        def callback(study, trial):
            pbar.update(1)
            # Log information about all trials
            logging.info("")
            logging.info("Optuna Trials:")
            for trial in study.trials:
                logging.info("")
                logging.info("Trial number: %d", trial.number)
                logging.info("  Params: %s", trial.params)
                logging.info("  Value: %s", trial.value)
                #logging.info("  State: %s", trial.state)
                logging.info("") 
            # Log information about the best trial after each trial
            logging.info('Best trial so far: %s', study.best_trial.params)
            logging.info(' with evaluation value: %s', study.best_trial.value)
            logging.info(' which is the trial nr. %s', study.best_trial.number)
            logging.info('')
            logging.info("")
                
        study.optimize(objective_wrapper(args), n_trials=n_trials, callbacks=[callback], n_jobs=n_jobs)
    
    print('Best evaluation values:', study.best_trial.values)
    print('Best trial:', study.best_trial.params)
    logging.info('Best trial overall: %s', study.best_trial.params)
    logging.info('with (Best) evaluation value overall: %s', study.best_trial.values)
    logging.info("")
    return study.best_trial.params, study.best_trial.number

# To create a resumable Optuna study
def run_optuna_optimization(args, n_trials=10, n_jobs=1):

    
    # Create or load the Study object named
    name_of_study = 'study_by_me'
    
    # Define the SQLite storage backend
    study_storage = 'sqlite:///output_of_model/optuna_study_storage.db'
    
    """
    Note that the storage doesn’t store the state of the instance of samplers and pruners. When we resume a study with a sampler whose 
    seed  argument is specified for reproducibility, you need to restore the sampler with using pickle.
    
    Samplers are responsible for generating parameter configurations (i.e., trials) during the optimization process, 
    and pruners are responsible for early stopping of trials based on certain conditions.
    """
    
    try:
        restored_sampler = pickle.load(open("output_of_model/optuna_sampler.pkl", "rb"))
        print('Loading study sampler!')
    except:
        restored_sampler = None
    
    # Suppress log messages of Optuna     
    #optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Load existing study or create a new one
    study = optuna.create_study(
        direction='maximize', study_name=name_of_study, storage=study_storage, load_if_exists=True, sampler=restored_sampler)
    
    # Define the callback function to log trial information
    def callback(study, trial):
        logging.info("")
        logging.info("Optuna Trials:")
        for trial in study.trials:
            logging.info("")
            logging.info("Trial number: %d", trial.number)
            logging.info("  Params: %s", trial.params)
            logging.info("  Value: %s", trial.value)
            logging.info("") 
        logging.info('Best trial so far: %s', study.best_trial.params)
        logging.info(' with evaluation value: %s', study.best_trial.value)
        logging.info(' which is the trial nr. %s', study.best_trial.number)
        logging.info('')
        logging.info("")
    
    # Run the optimization process
    with tqdm(total=n_trials) as pbar:
        def pbar_callback(study, trial):
            pbar.update(1)
            callback(study, trial)
        
        study.optimize(objective_wrapper(args), n_trials=n_trials, callbacks=[pbar_callback], n_jobs=n_jobs)
    
    # Save the study state
    study.trials_dataframe().to_csv('output_of_model/optuna_study_state.csv')
    
    # Save the sampler with pickle to be loaded later.
    with open("output_of_model/optuna_sampler.pkl", "wb") as fout:
        pickle.dump(study.sampler, fout)
    
    # Print and log information about the best trial
    print('Best evaluation values:', study.best_trial.value)
    print('Best trial:', study.best_trial.params)
    logging.info('Best trial overall: %s', study.best_trial.params)
    logging.info('with (Best) evaluation value overall: %s', study.best_trial.value)
    logging.info("")
    
    return study.best_trial.params, study.best_trial.number


