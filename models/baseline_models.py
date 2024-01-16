import random
from sklearn import linear_model, neighbors, svm, tree, ensemble
from sklearn.linear_model import BayesianRidge, ElasticNet
from sklearn.naive_bayes import GaussianNB
from models.basemodel import BaseModel

'''
    Define all Models implemented by the Sklearn library: 
    Linear Model, KNN, SVM, Decision Tree, Random Forest
'''

'''
    Linear Model - Ordinary least squares Linear Regression / Logistic Regression
    
    Takes no hyperparameters
'''

# class LinearModel(BaseModel):

#     def __init__(self, params, args):
#         super().__init__(params, args)

#         if args.objective == "regression":
#             if args.model_name =='LassoLinearModel':
#               self.model = linear_model.Lasso(alpha=params['alpha'])

#             elif args.model_name =='RidgeLinearModel':
#               self.model = linear_model.Ridge(alpha=params['alpha'])

#             else:
#               self.model = linear_model.LinearRegression(n_jobs=-1)
            
#         elif args.objective == "classification":
#             self.model = linear_model.LogisticRegression(multi_class="multinomial", n_jobs=-1)
#         elif args.objective == "binary":
#             self.model = linear_model.LogisticRegression(n_jobs=-1)

#     @classmethod
#     def define_trial_parameters(cls, trial, args):
#         if args.model_name == 'LassoLinearModel' or args.model_name == 'RidgeLinearModel':
#           params = {
#             "alpha" : trial.suggest_float("alpha", 0.01, 10.0)
#           }
#         else:
#           params = dict()
#         return params


class LinearModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        if args.objective == "regression":
            if args.model_name == 'ElasticNet':
                self.model = ElasticNet(alpha=params['alpha'], l1_ratio=params['l1_ratio'])
            else:
                self.model = linear_model.LinearRegression(n_jobs=-1)
            
        elif args.objective == "classification":
            self.model = linear_model.LogisticRegression(multi_class="multinomial", n_jobs=-1)
        elif args.objective == "binary":
            self.model = linear_model.LogisticRegression(n_jobs=-1)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        if args.model_name == 'ElasticNet':
            params = {
                "alpha": trial.suggest_float("alpha", 0.01, 10.0),
                "l1_ratio":  trial.suggest_float("l1_ratio", 0.0, 1.0)
            }
        else:
            params = dict()
        return params

'''
For regression tasks, it uses the BayesianRidge model from sklearn, which is a type of Bayesian linear regression.

For classification or binary tasks, it uses the GaussianNB model from sklearn, which is a Gaussian Naive Bayes model.
'''

class BayesianModel(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = BayesianRidge()
        elif args.objective == "classification" or args.objective == "binary":
            self.model = GaussianNB()

    def fit(self, X, y, X_val=None, y_val=None):
        return super().fit(X, y, X_val, y_val)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = dict()
        return params

'''
    K-Neighbors Regressor - Regression/Classification based on k-nearest neighbors
    
    Takes number of neighbors as hyperparameters
'''


class KNN(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = neighbors.KNeighborsRegressor(n_neighbors=params["n_neighbors"], n_jobs=-1)
        elif args.objective == "classification" or args.objective == "binary":
            self.model = neighbors.KNeighborsClassifier(n_neighbors=params["n_neighbors"], n_jobs=-1)

    def fit(self, X, y, X_val=None, y_val=None):
        max_samples = 10000
        if X.shape[0] > max_samples:
            idx = random.sample(list(range(X.shape[0])), max_samples)
            X = X[idx]
            y = y[idx]

        return super().fit(X, y, X_val, y_val)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "n_neighbors": trial.suggest_categorical("n_neighbors", list(range(3, 42, 2)))
        }
        return params


'''
    Support Vector Machines - Epsilon-Support Vector Regression / C-Support Vector Classification
    
    Takes the regularization parameter as hyperparameter
'''


class SVM(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            kernel = 'sigmoid' if args.num_features < 5 else 'rbf'
            print('Kernel: ', kernel)
            self.model = svm.SVR(C=params["C"], kernel=kernel)
        elif args.objective == "classification" or args.objective == "binary":
            self.model = svm.SVC(C=params["C"], probability=True)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "C": trial.suggest_float("C", 1e-10, 1e10, log=True)
        }
        return params


'''
    Decision Tree - Decision Tree Regressor/Classifier
    
    Takes the maximum depth of the tree as hyperparameter
'''


class DecisionTree(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = tree.DecisionTreeRegressor(max_depth=params["max_depth"])
        elif args.objective == "classification" or args.objective == "binary":
            self.model = tree.DecisionTreeClassifier(max_depth=params["max_depth"])

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True)
        }
        return params


'''
    Random Forest - Random Forest Regressor/Classifier
    
    Takes the maximum depth of the trees and the number of estimators as hyperparameter
'''


class RandomForest(BaseModel):

    def __init__(self, params, args):
        super().__init__(params, args)

        if args.objective == "regression":
            self.model = ensemble.RandomForestRegressor(n_estimators=params["n_estimators"],
                                                        max_depth=params["max_depth"], n_jobs=-1)
        elif args.objective == "classification" or args.objective == "binary":
            self.model = ensemble.RandomForestClassifier(n_estimators=params["n_estimators"],
                                                         max_depth=params["max_depth"], n_jobs=-1)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 12, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 5, 100, log=True)
        }
        return params
