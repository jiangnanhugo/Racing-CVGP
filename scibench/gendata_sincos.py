import os

from symbolic_equation_evaluator_public import decrypt_equation, Equation_evaluator
import numpy as np
from symbolic_data_generator import *


def to_csv(X, y, filename):
    d = np.concatenate((X, y), axis=1)
    np.random.shuffle(d)
    np.savetxt(filename + ".csv", d, delimiter=",")


if __name__ == '__main__':
    basepath = "/home/jiangnan/PycharmProjects/scibench/data/unencrypted/equations_trigometric/sincos_nv8_nt812_prog_{}.in"
    to_folder = "/home/jiangnan/PycharmProjects/scibench/eureqa/data/equations_trigometric/sincos_nv8_nt812_prog_{}"
    for prog in range(10):
        filename = basepath.format(prog)
        data_query_oracle = Equation_evaluator(filename, noise_type='normal', noise_scale=0.0, metric_name='neg_mse')
        dataX = DataX(data_query_oracle.get_vars_range_and_types())
        batchsize = 100000

        # X = np.random.rand(batchsize, n_input) * 9.5 + 0.5
        X = dataX.randn(sample_size=batchsize)
        y = data_query_oracle.evaluate(X).reshape(-1,1)
        print(X.shape, y.shape)
        filename_csv = to_folder.format(prog)
        to_csv(X, y, filename_csv)
        print(f"{prog} done......")
        # print(one_eq['eq_expression'].execute(X))
