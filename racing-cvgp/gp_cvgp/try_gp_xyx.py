from library import Library
import argparse
from program import Program
import regress_task
from const import ScipyMinimize
from symbolic_data_generator import *
from symbolic_equation_evaluator_public import Equation_evaluator
from functions import create_tokens
import gp_xyx

import numpy as np
import random
import time

averaged_var_y = 10  # 569

config = {
    'neg_mse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.01},
    'neg_nmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.01 / averaged_var_y},
    'neg_nrmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': np.sqrt(0.01 / averaged_var_y)},
    'neg_rmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': 0.1},
    'inv_mse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': -1 / (1 + 0.01)},
    'inv_nmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': -1 / (1 + 0.01 / averaged_var_y)},
    'inv_nrmse': {'expr_consts_thres': 1e-3, 'expr_obj_thres': -1 / (1 + np.sqrt(0.01 / averaged_var_y))},
}


def run_expanding_gp(equation_name, metric_name, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    dataXgen = DataX(data_query_oracle.get_vars_range_and_types())
    nvar = data_query_oracle.get_nvars()

    regress_batchsize = 256
    opt_num_expr = 5

    expr_obj_thres = data_query_oracle.expr_obj_thres
    expr_consts_thres = config[metric_name]['expr_consts_thres']

    # gp parameters
    cxpb = 0.8
    mutpb = 0.8
    maxdepth = 2
    tour_size = 3
    hof_size = 50 #0

    population_size = 100
    n_generations = 20

    # get all the functions and variables ready
    all_tokens = create_tokens(nvar, data_query_oracle.function_set, protected=True)
    protected_library = Library(all_tokens)

    protected_library.print_library()

    # get program ready
    Program.library = protected_library
    Program.opt_num_expr = opt_num_expr
    Program.expr_obj_thres = expr_obj_thres
    Program.expr_consts_thres = expr_consts_thres

    Program.set_execute(True)  # protected = True

    # set const_optimizer
    Program.const_optimizer = ScipyMinimize()
    Program.noise_std = noise_scale

    # set the task
    allowed_input_tokens = np.zeros(nvar, dtype=np.int32)  # set it for now. Will change in gp.run
    Program.task = regress_task.RegressTaskV1(regress_batchsize,
                                              allowed_input_tokens,
                                              dataXgen,
                                              data_query_oracle)

    # set gp helper
    gp_helper = gp_xyx.GPHelper()
    gp_helper.library = protected_library

    # set GP
    gp_xyx.ExpandingGeneticProgram.library = protected_library
    gp_xyx.ExpandingGeneticProgram.gp_helper = gp_helper
    egp = gp_xyx.ExpandingGeneticProgram(cxpb, mutpb, maxdepth, population_size,
                                         tour_size, hof_size, n_generations, nvar)

    # run GP
    egp.run()

    # print
    print('final hof=')
    egp.print_hof()
    print('egp.timer_log=', egp.timer_log)


def run_gp(equation_name, metric_name, noise_type, noise_scale):
    data_query_oracle = Equation_evaluator(equation_name, noise_type, noise_scale, metric_name)
    temp = data_query_oracle.get_vars_range_and_types()
    dataXgen = DataX(temp)
    nvar = data_query_oracle.get_nvars()

    regress_batchsize = 256
    opt_num_expr = 1  # currently do not need to re-run the experiments multiple times.

    # gp parameters
    cxpb = 0.8
    mutpb = 0.8
    maxdepth = 2
    population_size = 500  # 00
    tour_size = 3
    hof_size = 100
    n_generations = 80  # 00

    # get all the functions and variables ready
    all_tokens = create_tokens(nvar, data_query_oracle.function_set, protected=True)
    protected_library = Library(all_tokens)

    protected_library.print_library()

    # everything is allowed.
    allowed_input_tokens = np.ones(nvar, dtype=np.int32)
    protected_library.set_allowed_input_tokens(allowed_input_tokens)

    # get program ready
    Program.library = protected_library
    Program.opt_num_expr = opt_num_expr
    Program.set_execute(True)  # protected = True

    # set const_optimizer
    Program.const_optimizer = ScipyMinimize()
    Program.noise_std = noise_scale

    # set the task
    Program.task = regress_task.RegressTaskV1(regress_batchsize,
                                              allowed_input_tokens,
                                              dataXgen,
                                              data_query_oracle)

    # set gp helper
    gp_helper = gp_xyx.GPHelper()
    gp_helper.library = protected_library

    # set GP
    gp_xyx.GeneticProgram.library = protected_library
    gp_xyx.GeneticProgram.gp_helper = gp_helper
    gp = gp_xyx.GeneticProgram(cxpb, mutpb, maxdepth, population_size, tour_size, hof_size, n_generations)

    # run GP
    gp.run()

    # print
    print('final hof=')
    gp.print_hof()
    print('gp.timer_log=', gp.timer_log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--equation_name", help="the filename of the true program.")
    parser.add_argument("--metric_name", type=str, help="The name of the metric for loss.")
    parser.add_argument("--noise_type", type=str, help="The name of the noises.")
    parser.add_argument("--expr_obj_thres", type=float, help="Threshold")
    parser.add_argument("--noise_scale", type=float, default=0.0, help="This parameter adds the standard deviation of the noise")
    parser.add_argument("--expand_gp", action="store_true", help="whether run normal gp (expand_gp=False) or expand_gp.")

    args = parser.parse_args()

    seed = int(time.perf_counter() * 10000) % 1000007
    random.seed(seed)
    print('random seed=', seed)

    seed = int(time.perf_counter() * 10000) % 1000007
    np.random.seed(seed)
    print('np.random seed=', seed)

    if args.expand_gp:
        run_expanding_gp(args.equation_name, args.metric_name, args.noise_type, args.noise_scale)
    else:
        run_gp(args.equation_name, args.metric_name, args.noise_type, args.noise_scale)
