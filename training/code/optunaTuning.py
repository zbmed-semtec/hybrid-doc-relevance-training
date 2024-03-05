import os
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


def objective_wrapper(args):
    def objective(trial):
        # Suggest hyperparameters for Doc2Vec
        vector_size = trial.suggest_int('vector_size', 100, 600, step=5)
        window = trial.suggest_int('window', 5, 25)
        min_count = trial.suggest_int('min_count', 1, 6)
        epochs = trial.suggest_int('epochs', 5, 15)
        workers = trial.suggest_int('workers', 2, 8)
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
        # model_directory_trial = os.path.join(model_directory, f"Model_{trial.number}.model")
        # permissions = 0o755  # This sets permissions to rwxr-xr-x
        # os.chmod(model_directory, permissions)

        # Assume run() trains the model and returns the path to a file with similarity scores
        similarity_file = run(params, args, tuning = True, save_model=False)
        
        ref_pmids, data = precision.read_file(similarity_file)
        vector = precision.generate_vector(ref_pmids, data)

        precision_5 = list(np.mean(vector, axis=0).round(4))

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


