import argparse
from dso.program import Program
import pandas as pd
# used to load true program for data generation
from dso.library import Library, Token, PlaceholderConstant
from dso.const import ScipyMinimize
import dso.functions as functions
import pickle
import numpy as np

from sympy.parsing.sympy_parser import parse_expr
import scipy
from sklearn.metrics import r2_score
from symbolic_data_generator import DataX
from symbolic_equation_evaluator_public import Equation_evaluator


def read_dso_expression(csv_file, X_test):
    df = pd.read_csv(csv_file)
    # print(df.head())
    expression = df['expression'].iloc[0][1:-1]
    expr = parse_expr(expression)
    print('dso', expr.expand())
    var_x = expr.free_symbols
    print(var_x)
    y_hat = np.zeros(X_test.shape[0])
    for idx in range(X_test.shape[0]):
        X = X_test[idx, :]
        val_dict = {}
        for x in var_x:
            i = int(x.name[1:]) - 1
            val_dict[x] = X[i]
        y_hat[idx] = expr.evalf(subs=val_dict)

    return y_hat


def load_true_program(equation_name, metric_name, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()


def compute_dso_all_metrics(equation_name, noise_type, noise_scale, csv_file, testset_size, metric_name="neg_mse"):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()

    X_test = dataXgen.randn(testset_size)
    y_test = data_query_oracle.evaluate(X_test)

    # Compute predictions on test data
    y_hat = read_dso_expression(csv_file, X_test)

    ## add all metrics

    print('%' * 30)
    dict_of_rs = {}
    for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
        metric_params = (1.0,)
        metric = make_regression_metric(metric_name, *metric_params)
        r = metric(y_test, y_hat, np.var(y_test))
        dict_of_rs[metric_name] = r
        # print('{} {}'.format(metric_name, r))

    for metric_name in ['neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
        metric_params = [1.0, ]
        metric = make_regression_metric(metric_name, *metric_params)
        r = metric(y_test, y_hat)
        # print('{} {}'.format(metric_name, r))
        dict_of_rs[metric_name] = r
    # dict_of_rs['r2_score'] = r2_score(y_test, y_hat)
    return dict_of_rs


def compute_eureqa_all_metrics(equation_filename, noise_type, noise_scale, expr_str, testset_size, metric_name=""):
    data_query_oracle = Equation_evaluator(equation_filename, noise_type, noise_scale, metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()

    X_test = dataXgen.randn(testset_size)
    y_test = data_query_oracle.evaluate(X_test)
    # y_test_noiseless = y_test
    expr_str = expr_str.replace("^", "**")
    print("orig expr string:", expr_str)
    expr = parse_expr(expr_str)
    print('eureqa', expr.expand())
    var_x = expr.free_symbols
    print(var_x)
    y_hat = np.zeros(X_test.shape[0])
    for idx in range(X_test.shape[0]):
        X = X_test[idx, :]
        val_dict = {}
        for x in var_x:
            i = int(x.name[1:]) - 1
            val_dict[x] = X[i]
        y_hat[idx] = expr.evalf(subs=val_dict)
    print('%' * 30)
    dict_of_rs = {}
    for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
        metric_params = (1.0,)
        metric = make_regression_metric(metric_name, *metric_params)
        r = metric(y_test, y_hat, np.var(y_test))
        dict_of_rs[metric_name] = r

    for metric_name in ['neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
        metric_params = [1.0, ]
        metric = make_regression_metric(metric_name, *metric_params)
        r = metric(y_test, y_hat)
        dict_of_rs[metric_name] = r

    return dict_of_rs


def make_regression_metric(name, *args):
    """
    Factory function for a regression metric. This includes a closures for
    metric parameters and the variance of the training data.

    Parameters
    ----------

    name : str
        Name of metric. See all_metrics for supported metrics.

    args : args
        Metric-specific parameters

    Returns
    -------

    metric : function
        Regression metric mapping true and estimated values to a scalar.

    invalid_reward: float or None
        Reward value to use for invalid expression. If None, the training
        algorithm must handle it, e.g. by rejecting the sample.

    max_reward: float
        Maximum possible reward under this metric.
    """

    all_metrics = {

        # Negative mean squared error
        # Range: [-inf, 0]
        # Value = -var(y) when y_hat == mean(y)
        "neg_mse": (lambda y, y_hat: -np.mean((y - y_hat) ** 2),
                    0),

        # Negative root mean squared error
        # Range: [-inf, 0]
        # Value = -sqrt(var(y)) when y_hat == mean(y)
        "neg_rmse": (lambda y, y_hat: -np.sqrt(np.mean((y - y_hat) ** 2)),
                     0),

        # Negative normalized mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nmse": (lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,
                     0),

        # Negative normalized root mean squared error
        # Range: [-inf, 0]
        # Value = -1 when y_hat == mean(y)
        "neg_nrmse": (lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y),
                      0),

        # (Protected) negative log mean squared error
        # Range: [-inf, 0]
        # Value = -log(1 + var(y)) when y_hat == mean(y)
        "neglog_mse": (lambda y, y_hat: -np.log(1 + np.mean((y - y_hat) ** 2)),
                       0),

        # (Protected) inverse mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]*var(y)) when y_hat == mean(y)
        "inv_mse": (lambda y, y_hat: 1 / (1 + np.mean((y - y_hat) ** 2)),
                    1),

        # (Protected) inverse normalized mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nmse": (lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2) / var_y),
                     1),

        # (Protected) inverse normalized root mean squared error
        # Range: [0, 1]
        # Value = 1/(1 + args[0]) when y_hat == mean(y)
        "inv_nrmse": (lambda y, y_hat, var_y: 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),
                      1),

        # Fraction of predicted points within p0*abs(y) + p1 band of the true value
        # Range: [0, 1]
        "fraction": (lambda y, y_hat: np.mean(abs(y - y_hat) < args[0] * abs(y) + args[1]),
                     2),

        # Pearson correlation coefficient
        # Range: [0, 1]
        "pearson": (lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],
                    0),

        # Spearman correlation coefficient
        # Range: [0, 1]
        "spearman": (lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
                     0)
    }

    assert name in all_metrics, "Unrecognized reward function name."
    # assert len(args) == all_metrics[name][1], "For {}, expected {} reward function parameters; received {}.".format(name,all_metrics[name][1], len(args))
    metric = all_metrics[name][0]

    return metric



