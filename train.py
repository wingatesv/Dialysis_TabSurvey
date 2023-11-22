import logging
import sys
import numpy as np
import optuna

from models import str2model
from utils.load_data import load_data
from utils.scorer import get_scorer
from utils.timer import Timer
from utils.io_utils import save_results_to_file, save_hyperparameters_to_file, save_loss_to_file
from utils.parser import get_parser, get_given_parameters_parser

from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

def augment_data(X_train, y_train, augmentation_params, apply_gaussian_noise = False, apply_random_jitter = True):
    # Initialize augmented data as original data
    X_train_augmented = X_train
    y_train_augmented = y_train

    # Check if Gaussian noise should be applied
    if apply_gaussian_noise:
        print('Gaussian Noise Level: ', augmentation_params['gaussian_noise_level'])
        # Perform data augmentation by adding Gaussian noise to the features (X)
        noise = np.random.normal(loc=0, scale=augmentation_params['gaussian_noise_level'], size=X_train.shape)
        X_train_augmented_noise = X_train + noise
        # Combine the original features with the augmented features
        X_train_augmented = np.vstack([X_train_augmented, X_train_augmented_noise])
        y_train_augmented = np.hstack([y_train_augmented, y_train])

    # Check if random jitter should be applied
    if apply_random_jitter:
        print('Jitter Level: ', augmentation_params['jitter_level'])
        # Perform data augmentation by adding random jittering to the features (X)
        jitter = np.random.uniform(-augmentation_params['jitter_level'], augmentation_params['jitter_level'], size=X_train.shape)
        X_train_augmented_jitter = X_train + jitter
        # Combine the original features with the augmented features
        X_train_augmented = np.vstack([X_train_augmented, X_train_augmented_jitter])
        y_train_augmented = np.hstack([y_train_augmented, y_train])

    return X_train_augmented, y_train_augmented


# def augment_data(X_train, y_train, augmentation_params):
#     print('Gaussian Noise Level: ', augmentation_params['gaussian_noise_level'], ' Jitter Level: ', augmentation_params['jitter_level'])

#     # Perform data augmentation by adding Gaussian noise to the features (X)
#     noise = np.random.normal(loc=0, scale=augmentation_params['gaussian_noise_level'], size=X_train.shape)
#     X_train_augmented_noise = X_train + noise

#     # Perform data augmentation by adding random jittering to the features (X)
#     jitter = np.random.uniform(-augmentation_params['jitter_level'], augmentation_params['jitter_level'], size=X_train.shape)
#     X_train_augmented_jitter = X_train + jitter

#     # Combine the original features with the augmented features
#     X_train = np.vstack([X_train, X_train_augmented_noise, X_train_augmented_jitter])

#     # Generate new target values for the augmented samples
#     y_train_augmented = np.hstack([y_train, y_train, y_train])

#     return X_train, y_train_augmented


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

        # Perform data augmentation on regression data
        if args.regression_aug:
          X_train, y_train = augment_data(X_train, y_train, augmentation_params)

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
              'gaussian_noise_level': trial.suggest_float("gaussian_noise_level", 0.01, 0.5),
              'jitter_level': trial.suggest_float("jitter_level", 0.01, 0.5)
          }

          # Include augmentation_params into trial_params
          trial_params.update(augmentation_params)

        # Create model
        model = self.model_name(trial_params, self.args)

        # Cross validate the chosen hyperparameters
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
          'gaussian_noise_level': study.best_trial.params['gaussian_noise_level'],
          'jitter_level': study.best_trial.params['jitter_level']
      }
    cross_validation(model, X, y, args, best_augmentation_params, save_model=True)


def main_once(args):
    print("Train model with given hyperparameters")
    X, y = load_data(args)

    model_name = str2model(args.model_name)

    parameters = args.parameters[args.dataset][args.model_name]
    model = model_name(parameters, args)

    sc, time = cross_validation(model, X, y, args)
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
