import time

import numpy as np
import scipy
from scibench.encrypted_equations import physic_equations

import json
import pickle
from sympy.parsing.sympy_parser import parse_expr

# call a million batch of dataset. compute the time.
# a class takes the input of a file, that a file is an equation.
# the class will return a batch of data, everytime it was queried.
# don't do the tcp version.
# create a offline version to bitbucket.org
#
# future competition.
# offline evaluation: that are not open.
# type of noise, rate of noise.

eq_name_dict = {
    'hash_code': "sincosinv/prog_0",
}


# init
# 1nd way: the `eq_filename` that contains the equation
# 2nd way is for compeition: check `initlizer_debug`, the file is encrpted.

PRIVATE_KEY="9endsfiosudewdcx98ewds!"

class Equation_evaluator(object):
    def __init__(self, eq_filename_hashed, initlizer_debug=False, noise_type='normal', noise_scale=0.1, metric_name="neg_nmse"):
        '''
        true_program: the program to map from X to Y
        batch_size: number of data points.
        noise_type, noise_scale: the type and scale of noise.
        metric_name: evaluation metric name for `y_true` and `y_pred`
        '''
        assert dataset_family in ['feynman', 'trigonometric'], "the dataset family not found!"
        self.true_equation = None
        assert initlizer_debug==True, ""
        if initlizer_debug == False:
            self.load_true_equation(eq_name_dict[eq_filename_hashed])

        # metric
        self.metric_name = metric_name
        self.metric = make_regression_metric(metric_name)

        # noise
        assert noise_type in ['uniform', 'normal', 'exponential', 'laplace', 'logistic'], f"the noise_type: {noise_type} not defined"
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noises = construct_noise(self.noise_type)

    def random_choose_equation(self):
        # 1. rancomly choose 1 in eq_name_dict
        # 2. call the hashlib.md5(eq_nbame)
        # 3. return the md5string
        # return hashlib.md5()
        pass

    def get_nvars(self):
        return self.true_equation.get_nvars()

    def get_function_ops(self):
        return self.true_equation.get_ops()

    def load_true_equation(self):
        raise NotImplementedError("true equation is not loaded!")

    def evaluate(self, X):
        """
        evaluate the y_true from given input X
        """
        nvar, batch_size = X.shape
        if self.true_equation is None:
            self.load_true_equation()
        y_true = self.true_equation.execute(X) + self.noises(self.noise_scale, batch_size)
        return y_true

    def compute_metric_loss(self, y_true, y_pred):
        """
        evaluate the metric value between y_true and y_pred
        """
        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            loss = self.metric(y_true, y_pred, np.var(y_true))
        elif self.metric_name in ['neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
            loss = self.metric(y_true, y_pred)
        return loss


class Feynman_evaluator(Equation_evaluator):
    def __init__(self, dataset_family, eq_name, noise_type, noise_scale, metric_name):
        super.__init__(dataset_family, eq_name, noise_type, noise_scale, metric_name)

    def load_true_equation(self):
        self.true_equation = physic_equations.get_eq_obj(self.eq_name)


class Trigonometric_evaluator(Equation_evaluator):
    def __init__(self, dataset_family, eq_name, noise_type, noise_scale, metric_name):
        super.__init__(dataset_family, eq_name, noise_type, noise_scale, metric_name)
        equation = physic_equations.get_eq_obj(eq_name)

    def load_true_equation(self):
        pass
        # expr = parse_expr(expression_str)
        # var_x = expr.free_symbols


def read_picked_data(filename):
    return pickle.load(open(filename, 'rb'))


def construct_noise(noise_type):
    _all_samplers = {
        'normal': lambda scale, batch_size: np.random.normal(loc=0.0, scale=scale, size=batch_size),
        'exponential': lambda scale, batch_size: np.random.exponential(scale=scale, size=batch_size),
        'uniform': lambda scale, batch_size: np.random.uniform(low=-np.abs(scale), high=np.abs(scale), size=batch_size),
        'laplace': lambda scale, batch_size: np.random.laplace(loc=0.0, scale=scale, size=batch_size),
        'logistic': lambda scale, batch_size: np.random.logistic(loc=0.0, scale=scale, size=batch_size)
    }
    assert noise_type in _all_samplers, "Unrecognized noise_type" + noise_type

    return _all_samplers[noise_type]


def make_regression_metric(metric_name):
    """
    Factory function for a regression metric. This includes a closures for metric parameters and the variance of the training data.
    metric_name: Regression metric mapping true and estimated values to a scalar.
    """
    all_metrics = {
        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse": lambda y, y_hat: -np.mean((y - y_hat) ** 2),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse": lambda y, y_hat: -np.sqrt(np.mean((y - y_hat) ** 2)),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y),

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse": lambda y, y_hat: -np.log(1 + np.mean((y - y_hat) ** 2)),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse": lambda y, y_hat: 1 / (1 + np.mean((y - y_hat) ** 2)),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2) / var_y),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse": lambda y, y_hat, var_y: 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),

        # Pearson correlation coefficient       # Range: [0, 1]
        "pearson": lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],

        # Spearman correlation coefficient      # Range: [0, 1]
        "spearman": lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
        "accuracy(r2)": lambda y, y_hat: evaluate_accuracy_r2(y, y_hat)
    }

    assert metric_name in all_metrics, "Unrecognized reward function name."

    return all_metrics[metric_name]


def evaluate_accuracy_r2(y, y_hat, tau=0.95):
    from sklearn.metrics import r2_score
    score = r2_score(y, y_hat)
    return score
