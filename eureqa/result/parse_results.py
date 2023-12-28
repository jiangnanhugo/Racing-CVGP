import argparse
import os
import numpy as np
import pandas as pd
from sympy.parsing.sympy_parser import parse_expr
from symbolic_data_generator import DataX
from symbolic_equation_evaluator_public import Equation_evaluator


def compute_eureqa_all_metrics(equation_filename, noise_type, noise_scale, expr_str, testset_size, metric_name="neg_mse"):
    data_query_oracle = Equation_evaluator(equation_filename, noise_type, noise_scale, metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()

    X_test = dataXgen.randn(testset_size)
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
    dict_of_rs = data_query_oracle._evaluate_all_losses(X_test, y_hat)#data_query_oracle.(X, y_hat)

    return dict_of_rs


def parse_eureqa_solutions(eureqa_basepath, noise_type, noise_scale, is_numbered=True):
    df = pd.read_csv(eureqa_basepath)
    all_eureqa_r = {}
    result_dict = {}
    for i, row in df.iterrows():
        prog = row['benchmark']
        if is_numbered:
            idx = int(prog.split('_')[-1])
        else:
            idx=prog.split("/")[-1]
        predicted = row['solution']
        result_dict[idx] = predicted
        equreqa_ri = compute_eureqa_all_metrics(equation_filename=prog + '.in', expr_str=result_dict[idx], testset_size=256,
                                                noise_type=noise_type, noise_scale=noise_scale)
        all_eureqa_r[idx] = equreqa_ri
        # except:
        #     print(i, "eureqa cannot process")

    return all_eureqa_r


def pretty_print_eureqa(all_eureqa_rs, is_numbered):
    for key in ['neg_nmse', 'neg_mse', 'neg_rmse', 'neg_nrmse']:
        # print('{}\ndata idx, gp, expand_gp, dso'.format(key))
        print(key, ", EUREQA")
        if is_numbered:
            for idx in range(26):
                print(idx, end=", ")
                if idx in all_eureqa_rs:
                    print(all_eureqa_rs[idx][key])
                else:
                    print()
        else:
            for prog in all_eureqa_rs:
                print(prog, end=", ")
                if key in all_eureqa_rs[prog]:
                    print(all_eureqa_rs[prog][key])
                else:
                    print(",")
            print()


if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--metric', type=str, default='neg_mse')
    parser.add_argument('--noise_type', type=str, default="normal")
    parser.add_argument('--noise_scale', type=float, default=0.0)
    parser.add_argument('--eureqa_path', type=str, required=True)
    parser.add_argument('--is_numbered', type=int, default=0)

    # Parse the argument
    args = parser.parse_args()
    all_eureqa_r = parse_eureqa_solutions(args.eureqa_path, args.noise_type, args.noise_scale, args.is_numbered)
    print(all_eureqa_r)
    pretty_print_eureqa(all_eureqa_r,args.is_numbered)
