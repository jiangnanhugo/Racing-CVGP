import numpy as np


class RegressTaskV1(object):
    """
    input parameters:

    batchsize:
    allowed_input: 1 if the input can be in the approximated expr. 0 cannot.
    n_input: num of vars in X
    true_program: the program to map from X to Y

    reward_function(self, p) # in reward function need to decide on non-varying parameters

    evaluate(self, p)        # this is the inference task (evaluate the program on the test set).

    NOTE: nexpr should be left to program.optimize() (nexpr: number of experiments)
    """

    def __init__(self, batchsize, allowed_input, dataX, data_query_oracle):
        self.batchsize = batchsize
        self.allowed_input = allowed_input
        self.n_input = allowed_input.size
        self.dataX = dataX
        self.data_query_oracle = data_query_oracle

        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]
        self.X_fixed = np.random.rand(self.n_input)

    def set_allowed_inputs(self, allowed_inputs):
        self.allowed_input = np.copy(allowed_inputs)
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def set_allowed_input(self, i, flag):
        self.allowed_input[i] = flag
        self.fixed_column = [i for i in range(self.n_input) if self.allowed_input[i] == 0]

    def rand_draw_X_non_fixed(self):
        self.X = self.dataX.randn(sample_size=self.batchsize)

    def rand_draw_X_fixed(self):
        self.X_fixed = np.squeeze(self.dataX.randn(sample_size=1))

    def rand_draw_data_with_X_fixed(self):
        self.X = self.dataX.randn(sample_size=self.batchsize)
        if len(self.fixed_column):
            self.X[:, self.fixed_column] = self.X_fixed[self.fixed_column]


    def print_reward_function_all_metrics(self, p):
        """used for print the error for all metrics between the predicted program `p` and true program."""
        y_hat = p.execute(self.X)
        dict_of_result = self.data_query_oracle._evaluate_all_losses(self.X, y_hat)
        print('%' * 30)
        for mertic_name in dict_of_result:
            print(f"{mertic_name} {dict_of_result[mertic_name]}")
        print('%' * 30)

    def reward_function(self, p):
        y_hat = p.execute(self.X)
        return self.data_query_oracle._evaluate_loss(self.X, y_hat)
