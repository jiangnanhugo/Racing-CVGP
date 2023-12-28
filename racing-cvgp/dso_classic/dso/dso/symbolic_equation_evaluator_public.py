import os
from typing import List, Set
import scipy

import json
from cryptography.fernet import Fernet
from sympy import Symbol
from sympy import parse_expr

from fractions import Fraction
import numpy as np
import time

EQUATION_EXTENSION = ".in"


class Equation_evaluator(object):
    def __init__(self, true_equation_filename, noise_type='normal', noise_scale=0.0, metric_name="neg_nmse"):
        '''
        true_equation_filename: the program to map from X to Y
        noise_type, noise_scale: the type and scale of noise.
        metric_name: evaluation metric name for `y_true` and `y_pred`
        '''
        self.true_equation, self.num_vars, self.function_set, self.vars_range_and_types, self.expr = self.__load_equation(
            true_equation_filename)

        # metric
        self.metric_name = metric_name
        self.metric = make_regression_metric(metric_name)

        # noise
        self.noise_type = noise_type
        self.noise_scale = noise_scale
        self.noises = construct_noise(self.noise_type)
        self.start = time.time()

    # Declaring private method. This function cannot be called outside the class.
    def __load_equation(self, equation_name):
        self.eq_name = equation_name
        if not os.path.isfile(self.eq_name):
            print(self.eq_name, "file not found")
            exit()
            #raise FileNotFoundError("{} not found!".format(self.eq_name))

        one_equation = decrypt_equation(self.eq_name, key_filename="encrypted_equation/public.key")
        num_vars = int(one_equation['num_vars'])
        kwargs_list = [{'real': True} for _ in range(num_vars)]

        assert len(kwargs_list) == num_vars
        self.num_vars = num_vars
        x = [Symbol(f'X_{i}', **kwargs) for i, kwargs in enumerate(kwargs_list)]
        return one_equation['eq_expression'], int(one_equation['num_vars']), one_equation['function_set'], \
            one_equation['vars_range_and_types'], parse_expr(one_equation['expr'])

    def evaluate(self, X):
        # evaluate the y_true from given input X
        batch_size, nvar = X.shape
        assert self.num_vars == nvar, f"The number of variables in your input is {nvar}, but we expect {self.num_vars}"

        if self.true_equation is None:
            raise NotImplementedError('no equation is available')
        y_true = self.true_equation.execute(X) + self.noises(self.noise_scale, batch_size)
        # y_hat = self.get_symbolic_output(X) + self.noises(self.noise_scale, batch_size)
        # for y_i, y_hat_i in zip(y_true, y_hat):
        #     if np.abs(y_i - y_hat_i) > 1e-10:
        #         raise ArithmeticError(f'the difference are too large {y_i} {y_hat_i}')

        return y_true

    def evaluate_noiseless(self, X):
        # evaluate the y_true from given input X
        batch_size, nvar = X.shape
        assert self.num_vars == nvar, f"The number of variables in your input is {nvar}, but we expect {self.num_vars}"

        if self.true_equation is None:
            raise NotImplementedError('no equation is available')
        y_true = self.true_equation.execute(X)
        # y_hat = self.get_symbolic_output(X)
        # for y_i, y_hat_i in zip(y_true, y_hat):
        #     if np.abs(y_i - y_hat_i) > 1e-10:
        #         raise ArithmeticError(f'the difference are too large {y_i} {y_hat_i}')

        return y_true

    def get_symbolic_output(self, X_test):
        var_x = self.expr.free_symbols
        y_hat = np.zeros(X_test.shape[0])
        for idx in range(X_test.shape[0]):
            X = X_test[idx, :]
            val_dict = {}
            for x in var_x:
                i = int(x.name[2:])
                val_dict[x] = X[i]
            y_hat[idx] = self.expr.evalf(subs=val_dict)

        return y_hat

    def _evaluate_loss(self, X, y_pred):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_pred
        """
        y_true = self.evaluate(X)
        if self.metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            loss_val = self.metric(y_true, y_pred, np.var(y_true))
        elif self.metric_name in ['neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
            loss_val = self.metric(y_true, y_pred)
        return loss_val

    def _evaluate_all_losses(self, X, y_pred):
        """
        Compute the y_true based on the input X. And then evaluate the metric value between y_true and y_hat.
        Return a dictionary of all the loss values.
        """
        y_true = self.evaluate(X)
        loss_val_dict = {}
        metric_params = (1.0,)
        for metric_name in ['neg_nmse', 'neg_nrmse', 'inv_nrmse', 'inv_nmse']:
            metric = make_regression_metric(metric_name)
            loss_val = metric(y_true, y_pred, np.var(y_true))
            loss_val_dict[metric_name] = loss_val
        for metric_name in ['neg_mse', 'neg_rmse', 'neglog_mse', 'inv_mse']:
            metric = make_regression_metric(metric_name)
            loss_val = metric(y_true, y_pred)
            loss_val_dict[metric_name] = loss_val
        return loss_val_dict

    def _get_eq_name(self):
        return self.eq_name

    def get_vars_range_and_types(self):
        return self.vars_range_and_types

    def get_nvars(self):
        return self.num_vars

    def get_function_set(self):
        return self.function_set


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
    all_metrics = {
        # Negative mean squared error
        "neg_mse": lambda y, y_hat: -np.mean((y - y_hat) ** 2),
        # Negative root mean squared error
        "neg_rmse": lambda y, y_hat: -np.sqrt(np.mean((y - y_hat) ** 2)),
        # Negative normalized mean squared error
        "neg_nmse": lambda y, y_hat, var_y: -np.mean((y - y_hat) ** 2) / var_y,
        # Negative normalized root mean squared error
        "neg_nrmse": lambda y, y_hat, var_y: -np.sqrt(np.mean((y - y_hat) ** 2) / var_y),
        # (Protected) negative log mean squared error
        "neglog_mse": lambda y, y_hat: -np.log(1 + np.mean((y - y_hat) ** 2)),
        # (Protected) inverse mean squared error
        "inv_mse": lambda y, y_hat: 1 / (1 + np.mean((y - y_hat) ** 2)),
        # (Protected) inverse normalized mean squared error
        "inv_nmse": lambda y, y_hat, var_y: 1 / (1 + np.mean((y - y_hat) ** 2) / var_y),
        # (Protected) inverse normalized root mean squared error
        "inv_nrmse": lambda y, y_hat, var_y: 1 / (1 + np.sqrt(np.mean((y - y_hat) ** 2) / var_y)),
        # Pearson correlation coefficient       # Range: [0, 1]
        "pearson": lambda y, y_hat: scipy.stats.pearsonr(y, y_hat)[0],
        # Spearman correlation coefficient      # Range: [0, 1]
        "spearman": lambda y, y_hat: scipy.stats.spearmanr(y, y_hat)[0],
        # Accuracy based on R2 value.
        "accuracy(r2)": lambda y, y_hat: evaluate_accuracy_r2(y, y_hat)}
    assert metric_name in all_metrics, "Unrecognized reward function name."
    return all_metrics[metric_name]


def evaluate_accuracy_r2(y, y_hat, tau=0.95):
    from sklearn.metrics import r2_score
    score = r2_score(y, y_hat)
    return score


def decrypt_equation(eq_file, key_filename=None):
    with open(eq_file, 'rb') as enc_file:
        encrypted = enc_file.readline()
        if encrypted == b'1\n':
            encrypted = enc_file.readline()
            fernet = Fernet(open(key_filename, 'rb').read())
            decrypted = fernet.decrypt(encrypted)
        elif encrypted == b'0\n':
            decrypted = enc_file.readline()
    one_equation = json.loads(decrypted)
    preorder_traversal = eval(one_equation['eq_expression'])
    preorder_traversal = [tt[0] for tt in preorder_traversal]
    print(preorder_traversal)
    list_of_tokens = create_tokens(one_equation['num_vars'], one_equation['function_set'], protected=True)
    if 'pow' in preorder_traversal:
        list_of_tokens = list_of_tokens + [sciToken(np.power, "pow", arity=2, complexity=1)]
    protected_library = sciLibrary(list_of_tokens)

    sciProgram.library = protected_library
    sciProgram.set_execute(protected=True)
    #
    true_pr = build_program(preorder_traversal, protected_library)
    one_equation['eq_expression'] = true_pr
    print("-" * 20)
    for key in one_equation:
        print(key, "\t", one_equation[key])
    print("-" * 20)
    return one_equation


def build_program(preorder_traversal, library):
    preorder_actions = library.actionize(['const' if is_float(tok) else tok for tok in preorder_traversal])
    true_pr = sciProgram(preorder_actions)
    for loc, tok in enumerate(preorder_traversal):
        if is_float(tok):
            true_pr.traversal[loc] = PlaceholderConstant(tok)
    return true_pr


def is_float(s):
    """Determine whether the input variable can be cast to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


class sciToken(object):
    """
    An arbitrary token or "building block" of a Program object.

    """

    def __init__(self, function, name, arity, complexity, input_var=None):
        """
        name : str. Name of token.
        arity : int. Arity (number of arguments) of token.
        complexity : float. Complexity of token.
        function : callable. Function associated with the token; used for executable Programs.
        input_var : int or None. Index of input if this Token is an input variable, otherwise None.
        """
        self.function = function
        self.name = name
        self.arity = arity
        self.complexity = complexity
        self.input_var = input_var

        if input_var is not None:
            assert function is None, "Input variables should not have functions."
            assert arity == 0, "Input variables should have arity zero."

    def __call__(self, *args):
        """Call the Token's function according to input."""
        assert self.function is not None, "Token {} is not callable.".format(self.name)

        return self.function(*args)

    def __repr__(self):
        return self.name


class PlaceholderConstant(sciToken):
    """
    A Token for placeholder constants that will be optimized with respect to
    the reward function. The function simply returns the "value" attribute.

    Parameters
    ----------
    value : float or None
        Current value of the constant, or None if not yet set.
    """

    def __init__(self, value=None):
        if value is not None:
            value = np.atleast_1d(value)
        self.value = value
        super().__init__(function=self.function, name="const", arity=0, complexity=1)

    def function(self):
        assert self.value is not None, "Constant is not callable with value None."
        return self.value

    def __repr__(self):
        if self.value is None:
            return self.name
        return str(self.value[0])


class HardCodedConstant(sciToken):
    """
    A Token with a "value" attribute, whose function returns the value.
    """

    def __init__(self, value=None, name=None):
        """  Value of the constant. """
        assert value is not None, "Constant is not callable with value None. Must provide a floating point number or string of a float."
        assert is_float(value)
        value = np.atleast_1d(np.float32(value))
        self.value = value
        if name is None:
            name = str(self.value[0])
        super().__init__(function=self.function, name=name, arity=0, complexity=1)

    def function(self):
        return self.value


class sciLibrary(object):
    """
    Library of sciTokens. We use a list of sciTokens (instead of set or dict) since
    we so often index by integers given by the Controller.
    """

    def __init__(self, tokens):
        """
        tokens :List of available Tokens in the library.
        names : list of str, Names corresponding to sciTokens in the library.
        arities : list of int. Arities corresponding to sciTokens in the library.
        """

        self.tokens = tokens
        self.L = len(tokens)
        self.names = [t.name for t in tokens]
        self.arities = np.array([t.arity for t in tokens], dtype=np.int32)

    def print_library(self):
        print('============== LIBRARY ==============')
        print('{0: >8} {1: >10} {2: >8}'.format('ID', 'NAME', 'ARITY'))
        for i in range(self.L):
            print('{0: >8} {1: >10} {2: >8}'.format(i, self.names[i], self.arities[i]))

    def __getitem__(self, val):
        """Shortcut to get Token by name or index."""

        if isinstance(val, str):
            try:
                i = self.names.index(val)
            except ValueError:
                raise ModuleNotFoundError("sciToken {} does not exist.".format(val))
        elif isinstance(val, (int, np.integer)):
            i = val
        else:
            raise ModuleNotFoundError("sciLibrary must be indexed by str or int, not {}.".format(type(val)))

        try:
            token = self.tokens[i]
        except IndexError:
            raise ModuleNotFoundError("sciToken index {} does not exist".format(i))
        return token

    def tokenize(self, inputs):
        """Convert inputs to list of Tokens."""
        if isinstance(inputs, str):
            inputs = inputs.split(',')
        elif not isinstance(inputs, list) and not isinstance(inputs, np.ndarray):
            inputs = [inputs]
        tokens = [input_ if isinstance(input_, sciToken) else self[input_] for input_ in inputs]
        return tokens

    def actionize(self, inputs):
        """Convert inputs to array of 'actions', i.e. ints corresponding to Tokens in the Library."""
        tokens = self.tokenize(inputs)
        actions = np.array([self.tokens.index(t) for t in tokens], dtype=np.int32)
        return actions


GAMMA = 0.57721566490153286060651209008240243104215933593992


def logabs(x1):
    """Closure of log for non-positive arguments."""
    return np.log(np.abs(x1))


def expneg(x1):
    return np.exp(-x1)


def n2(x1):
    return np.power(x1, 2)


def n3(x1):
    return np.power(x1, 3)


def n4(x1):
    return np.power(x1, 4)


def n5(x1):
    return np.power(x1, 5)


def sigmoid(x1):
    return 1 / (1 + np.exp(-x1))


def harmonic(x1):
    if all(val.is_integer() for val in x1):
        return np.array([sum(Fraction(1, d) for d in range(1, int(val) + 1)) for val in x1], dtype=np.float32)
    else:
        return GAMMA + np.log(x1) + 0.5 / x1 - 1. / (12 * x1 ** 2) + 1. / (120 * x1 ** 4)


# Annotate unprotected ops
unprotected_ops = [
    # Binary operators
    sciToken(np.add, "add", arity=2, complexity=1),
    sciToken(np.subtract, "sub", arity=2, complexity=1),
    sciToken(np.multiply, "mul", arity=2, complexity=1),
    sciToken(np.power, "pow", arity=2, complexity=1),
    sciToken(np.divide, "div", arity=2, complexity=2),

    # Built-in unary operators
    sciToken(np.sin, "sin", arity=1, complexity=3),
    sciToken(np.cos, "cos", arity=1, complexity=3),
    sciToken(np.tan, "tan", arity=1, complexity=4),
    sciToken(np.exp, "exp", arity=1, complexity=4),
    sciToken(np.log, "log", arity=1, complexity=4),
    sciToken(np.sqrt, "sqrt", arity=1, complexity=4),
    sciToken(np.square, "n2", arity=1, complexity=2),
    sciToken(np.negative, "neg", arity=1, complexity=1),
    sciToken(np.abs, "abs", arity=1, complexity=2),
    sciToken(np.maximum, "max", arity=1, complexity=4),
    sciToken(np.minimum, "min", arity=1, complexity=4),
    sciToken(np.tanh, "tanh", arity=1, complexity=4),
    sciToken(np.reciprocal, "inv", arity=1, complexity=2),

    # Custom unary operators
    sciToken(logabs, "logabs", arity=1, complexity=4),
    sciToken(expneg, "expneg", arity=1, complexity=4),
    sciToken(n3, "n3", arity=1, complexity=3),
    sciToken(n4, "n4", arity=1, complexity=3),
    sciToken(n5, "n5", arity=2, complexity=3),
    sciToken(sigmoid, "sigmoid", arity=1, complexity=4),
    sciToken(harmonic, "harmonic", arity=1, complexity=4)
]

"""Define custom protected operators"""


def protected_div(x1, x2):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return np.where(np.abs(x2) > 0.001, np.divide(x1, x2), 1.)


def protected_exp(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 < 100, np.exp(x1), 0.0)


def protected_log(x1):
    """Closure of log for non-positive arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, np.log(np.abs(x1)), 0.)


def protected_sqrt(x1):
    """Closure of sqrt for negative arguments."""
    return np.sqrt(np.abs(x1))


def protected_inv(x1):
    """Closure of inverse for zero arguments."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(np.abs(x1) > 0.001, 1. / x1, 0.)


def protected_expneg(x1):
    with np.errstate(over='ignore'):
        return np.where(x1 > -100, np.exp(-x1), 0.0)


def protected_n2(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.square(x1), 0.0)


def protected_n3(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 3), 0.0)


def protected_n4(x1):
    with np.errstate(over='ignore'):
        return np.where(np.abs(x1) < 1e6, np.power(x1, 4), 0.0)


def protected_sigmoid(x1):
    return 1 / (1 + protected_expneg(x1))


# Annotate protected ops
protected_ops = [
    # Protected binary operators
    sciToken(protected_div, "div", arity=2, complexity=2),

    # Protected unary operators
    sciToken(protected_exp, "exp", arity=1, complexity=4),
    sciToken(protected_log, "log", arity=1, complexity=4),
    sciToken(protected_log, "logabs", arity=1, complexity=4),  # Protected logabs is support, but redundant
    sciToken(protected_sqrt, "sqrt", arity=1, complexity=4),
    sciToken(protected_inv, "inv", arity=1, complexity=2),
    sciToken(protected_expneg, "expneg", arity=1, complexity=4),
    sciToken(protected_n2, "n2", arity=1, complexity=2),
    sciToken(protected_n3, "n3", arity=1, complexity=3),
    sciToken(protected_n4, "n4", arity=1, complexity=3),
    sciToken(protected_sigmoid, "sigmoid", arity=1, complexity=4)
]

# Add unprotected ops to function map
function_map = {
    op.name: op for op in unprotected_ops
}

# Add protected ops to function map
function_map.update({
    "protected_{}".format(op.name): op for op in protected_ops
})

TERMINAL_TOKENS = set([op.name for op in function_map.values() if op.arity == 0])
UNARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 1])
BINARY_TOKENS = set([op.name for op in function_map.values() if op.arity == 2])


def create_tokens(n_input_var: int, function_set: List, protected) -> List:
    """
    Helper function to create Tokens.
    n_input_var : int. Number of input variable Tokens.
    function_set : list. Names of registered Tokens, or floats that will create new Tokens.
    protected : bool. Whether to use protected versions of registered Tokens.
    """

    tokens = []

    # Create input variable Tokens
    for i in range(n_input_var):
        token = sciToken(name="X_{}".format(i), arity=0, complexity=1, function=None, input_var=i)
        tokens.append(token)

    for op in function_set:
        # Registered Token
        if op in function_map:
            # Overwrite available protected operators
            if protected and not op.startswith("protected_"):
                protected_op = "protected_{}".format(op)
                if protected_op in function_map:
                    op = protected_op

            token = function_map[op]
        # Hard-coded floating-point constant
        elif op == 'const':
            token = PlaceholderConstant(1.0)
        else:
            raise ValueError("Operation {} not recognized.".format(op))

        tokens.append(token)

    return list(set(tokens))


class sciProgram(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarry (dtype: int)
        Array of integers whose values correspond to indices

    """
    task = None  # Task
    library = None  # Library
    execute = None  # Link to execute. Either cython or python

    def __init__(self, tokens=None):
        """
        Builds the Program from a list of of integers corresponding to Tokens.
        """
        # Can be empty if we are unpickling
        if tokens is not None:
            self._init(tokens)

    def _init(self, tokens):
        # pre-order of the program. the most important thing.
        self.traversal = [sciProgram.library[t] for t in tokens]

        # position of the constant
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]

        self.len_traversal = len(self.traversal)

        self.invalid = False  # always false.
        self.str = tokens.tostring()
        self.tokens = tokens

    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""

        execute_function = python_execute

        if protected:
            sciProgram.protected = True
            sciProgram.execute_function = execute_function
        else:
            sciProgram.protected = False

            class InvalidLog():
                """Log class to catch and record numpy warning messages"""

                def __init__(self):
                    self.error_type = None  # One of ['divide', 'overflow', 'underflow', 'invalid']
                    self.error_node = None  # E.g. 'exp', 'log', 'true_divide'
                    self.new_entry = False  # Flag for whether a warning has been encountered during a call to Program.execute()

                def write(self, message):
                    """This is called by numpy when encountering a warning"""

                    if not self.new_entry:  # Only record the first warning encounter
                        message = message.strip().split(' ')
                        self.error_type = message[1]
                        self.error_node = message[-1]
                    self.new_entry = True

                def update(self):
                    """If a floating-point error was encountered, set Program.invalid
                    to True and record the error type and error node."""

                    if self.new_entry:
                        self.new_entry = False
                        return True, self.error_type, self.error_node
                    else:
                        return False, None, None

            invalid_log = InvalidLog()
            np.seterrcall(invalid_log)  # Tells numpy to call InvalidLog.write() when encountering a warning

            # Define closure for execute function
            def unsafe_execute(traversal, X):
                """This is a wrapper for execute_function. If a floating-point error
                would be hit, a warning is logged instead, p.invalid is set to True,
                and the appropriate nan/inf value is returned. It's up to the task's
                reward function to decide how to handle nans/infs."""

                with np.errstate(all='log'):
                    y = execute_function(traversal, X)
                    invalid, error_node, error_type = invalid_log.update()
                    return y, invalid, error_node, error_type

            sciProgram.execute_function = unsafe_execute

    def execute(self, X):
        """
        Execute program on input X.

        Parameters:
        X : np.array. Input to execute the Program over.

        Returns
        result : np.array or list of np.array
            In a single-object Program, returns just an array.
        """
        if not sciProgram.protected:
            # return some weired error.
            result, self.invalid, self.error_node, self.error_type = sciProgram.execute_function(self.traversal, X)
        else:
            result = sciProgram.execute_function(self.traversal, X)
            # always protected. 1/div
        return result

    def print_expression(self):
        print("\tExpression {}: {}".format(0, self.traversal))

    def __repr__(self):
        """Prints the program's traversal"""
        return ','.join([repr(t) for t in self.traversal])


def python_execute(traversal, X):
    """
    Executes the program according to X using Python.

    Parameters
    ----------
    X : array-like, shape = [n_samples, n_features], where n_samples is the number of samples and n_features is the number of features.

    Returns
    -------
    y_hats : array-like, shape = [n_samples]
        The result of executing the program on X.
    """

    apply_stack = []

    for node in traversal:
        apply_stack.append([node])

        while len(apply_stack[-1]) == apply_stack[-1][0].arity + 1:
            token = apply_stack[-1][0]
            terminals = apply_stack[-1][1:]

            if token.input_var is not None:
                intermediate_result = X[:, token.input_var]
            else:
                intermediate_result = token(*terminals)
            if len(apply_stack) != 1:
                apply_stack.pop()
                apply_stack[-1].append(intermediate_result)
            else:
                return intermediate_result

    assert False, "Function should never get here!"
    return None
