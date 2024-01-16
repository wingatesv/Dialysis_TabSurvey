import logging
import sys
import numpy as np
import optuna
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import random 

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser
from utils.augmentation import mixup, cutmix, add_gaussian_noise, add_random_jitter




def dialysis_cross_validation(model, X, y, args, augmentation_params, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    # Get unique patient IDs
    patient_ids = X['patient ID'].unique()
  
    # Split the patient IDs into training and testing sets
    kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)

    for i, (train_index, test_index) in enumerate(kf.split(patient_ids)):

        train_patient_ids = patient_ids[train_index]
        test_patient_ids = patient_ids[test_index]
        # Split the patient IDs into training and testing sets
        train_patient_ids, val_patient_ids = train_test_split(train_patient_ids, test_size=0.1, random_state=args.seed)
        
        print('Number of Train patients: ',len(train_patient_ids))
        print('Number of Val patients: ',len(val_patient_ids))
        print('Number of Test patients: ',len(test_patient_ids))
        
        # Split the data into training and testing sets based on the patient IDs
        X_train = X[X['patient ID'].isin(train_patient_ids)].copy()
        X_val = X[X['patient ID'].isin(val_patient_ids)].copy()
        X_test = X[X['patient ID'].isin(test_patient_ids)].copy()
        
        y_train = y[y['patient ID'].isin(train_patient_ids)].copy()
        y_val = y[y['patient ID'].isin(val_patient_ids)].copy()
        y_test = y[y['patient ID'].isin(test_patient_ids)].copy()

        print('Train Set: ', X_train.shape)

        # Perform data augmentation on regression data
        if args.regression_aug:
          mixup_df = mixup(train_patient_ids, args.data_path, args.use_absorbance_only, augmentation_params)
          print('Mixup df shape: ',mixup_df.shape)

          cutmix_df = cutmix(train_patient_ids, args.data_path, args.use_absorbance_only, augmentation_params)
          print('Cutmix df shape: ',cutmix_df.shape)

          noise_df = add_gaussian_noise(train_patient_ids, args.data_path, args.target_variable, args.use_absorbance_only, augmentation_params)
          print('Noise df shape: ',noise_df.shape)

          jitter_df = add_random_jitter(train_patient_ids, args.data_path, args.target_variable, args.use_absorbance_only, augmentation_params)
          print('Jitter df shape: ',jitter_df.shape)

          X_train = pd.concat([X_train, mixup_df.drop([args.target_variable], axis=1), cutmix_df.drop([args.target_variable], axis=1), noise_df.drop([args.target_variable], axis=1), jitter_df.drop([args.target_variable], axis=1)])
          y_train = pd.concat([y_train, mixup_df[['patient ID', args.target_variable]], cutmix_df[['patient ID', args.target_variable]], noise_df[['patient ID', args.target_variable]], jitter_df[['patient ID', args.target_variable]]])
          print('Augmented Train Set: ', X_train.shape)
          print('Augmented y train Set: ', y_train.shape)

          # Get a list of unique patient IDs
          train_unique_patient_ids = X_train['patient ID'].unique()
          
          # Set the seed
          random.seed(args.seed)
          # Shuffle the patient IDs
          random.shuffle(train_unique_patient_ids)

          # Concatenate the data for each patient ID in the shuffled list
          X_train = pd.concat([X_train[X_train['patient ID'] == id] for id in train_unique_patient_ids])
          y_train = pd.concat([y_train[y_train['patient ID'] == id] for id in train_unique_patient_ids])

          X_train.to_csv('X_shuffled_train.csv', index=False)
          y_train.to_csv('y_shuffled_train.csv', index=False)


        # Remove the 'patient ID' column from the training and testing sets
        X_train.drop(['patient ID'], axis=1, inplace=True)
        X_val.drop(['patient ID'], axis=1, inplace=True)
        X_test.drop(['patient ID'], axis=1, inplace=True)

        y_train.drop(['patient ID'], axis=1, inplace=True)
        y_val.drop(['patient ID'], axis=1, inplace=True)
        y_test.drop(['patient ID'], axis=1, inplace=True)

        # Convert the training and testing sets to NumPy arrays
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values

        y_train = y_train.values.squeeze()
        y_val = y_val.values.squeeze()
        y_test = y_test.values.squeeze()

        print('Final X_train Shape: ', X_train.shape)
        print('Final y_train shape: ', y_train.shape)

        # Apply MinMaxScaler to the training and testing sets
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test  = scaler.transform(X_test)


        # Create a new unfitted version of the model
        curr_model = model.clone()
        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()


        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)

        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(f'Result of {i} fold',sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time (s):", round(train_timer.get_average_time(),4))
        print("Inference time (s):", round(test_timer.get_average_time(),4))

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc, (round(train_timer.get_average_time(),4), round(test_timer.get_average_time(),4))


def cross_validation(model, X, y, args, augmentation_params, save_model=False):
    # Record some statistics and metrics
    sc = get_scorer(args)
    train_timer = Timer()
    test_timer = Timer()

    if args.objective == "regression":
        kf = KFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    elif args.objective == "classification" or args.objective == "binary":
        kf = StratifiedKFold(n_splits=args.num_splits, shuffle=args.shuffle, random_state=args.seed)
    else:
        raise NotImplementedError("Objective" + args.objective + "is not yet implemented.")

    for i, (train_index, test_index) in enumerate(kf.split(X, y)):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=args.seed)

        # Create a new unfitted version of the model
        curr_model = model.clone()

        # Train model
        train_timer.start()
        loss_history, val_loss_history = curr_model.fit(X_train, y_train, X_val, y_val)
        train_timer.end()

        # Test model
        test_timer.start()
        curr_model.predict(X_test)
        test_timer.end()

        # Save model weights and the truth/prediction pairs for traceability
        curr_model.save_model_and_predictions(y_test, i)

        if save_model:
            save_loss_to_file(args, loss_history, "loss", extension=i)
            save_loss_to_file(args, val_loss_history, "val_loss", extension=i)

        # Compute scores on the output
        sc.eval(y_test, curr_model.predictions, curr_model.prediction_probabilities)

        print(f'Result of {i} fold',sc.get_results())

    # Best run is saved to file
    if save_model:
        print("Results:", sc.get_results())
        print("Train time (s):", round(train_timer.get_average_time(),4))
        print("Inference time (s):", round(test_timer.get_average_time(),4))

        # Save the all statistics to a file
        save_results_to_file(args, sc.get_results(),
                             train_timer.get_average_time(), test_timer.get_average_time(),
                             model.params)

    # print("Finished cross validation")
    return sc, (round(train_timer.get_average_time(),4), round(test_timer.get_average_time(),4))


class Objective(object):
    def __init__(self, args, model_name, X, y):
        # Save the model that will be trained
        self.model_name = model_name

        # Save the trainings data
        self.X = X
        self.y = y

        self.args = args

    def __call__(self, trial):
        # Define hyperparameters to optimize
        trial_params = self.model_name.define_trial_parameters(trial, self.args)


        # Initialize augmentation_params
        augmentation_params = dict()
        if self.args.regression_aug:
          augmentation_params = {
              'mixup_lambda': trial.suggest_categorical("mixup_lambda", [1/6, 2/6, 3/6, 4/6, 5/6]),
              'cutmix_lambda': trial.suggest_categorical("cutmix_lambda", [1/6, 2/6, 3/6, 4/6, 5/6]),
              'gaussian_noise_level': trial.suggest_float("gaussian_noise_level", 0.001, 0.5),
              'jitter_level': trial.suggest_float("jitter_level", 0.0001, 0.5)
          }

          # Include augmentation_params into trial_params
          trial_params.update(augmentation_params)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
        if self.args.dataset == 'Dialysis':
          sc, time = dialysis_cross_validation(model, self.X, self.y, self.args, augmentation_params)
        else:
          sc, time = cross_validation(model, self.X, self.y, self.args, augmentation_params)
        

        save_hyperparameters_to_file(self.args, trial_params, sc.get_results(), time)

        return sc.get_objective_result()


def main(args):
    print("Start hyperparameter optimization")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = args.model_name + "_" + args.dataset
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(direction=args.direction,
                                study_name=study_name,
                                storage=storage_name,
                                load_if_exists=True)
    study.optimize(Objective(args, model_name, X, y), n_trials=args.n_trials)
    print("Best parameters:", study.best_trial.params)

    # Run best trial again and save it!
    model = model_name(study.best_trial.params, args)

    best_augmentation_params = dict()
    if args.regression_aug:
      best_augmentation_params = {
          'mixup_lambda': study.best_trial.params['mixup_lambda'],
          'cutmix_lambda': study.best_trial.params['cutmix_lambda'],
          'gaussian_noise_level': study.best_trial.params['gaussian_noise_level'],
          'jitter_level': study.best_trial.params['jitter_level']
      }
    if args.dataset == 'Dialysis':
      dialysis_cross_validation(model, X, y, args, best_augmentation_params, save_model=True)
    else:
      cross_validation(model, X, y, args, best_augmentation_params, save_model=True)


def main_once(args):
    print("Train model with given hyperparameters")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)
    
    args.regression_aug = False
    augmentation_params = dict()
    
    sc, time = cross_validation(model, X, y, args, augmentation_params)
    print(f'Final Result of {args.num_splits}-Fold CV',sc.get_results())
    print('Training and Inference Time (s): ',time)


if __name__ == "__main__":
    parser = get_parser()
    arguments = parser.parse_args()
    print(arguments)

    if arguments.optimize_hyperparameters:
        main(arguments)
    else:
        # Also load the best parameters
        parser = get_given_parameters_parser()
        arguments = parser.parse_args()
        main_once(arguments)
