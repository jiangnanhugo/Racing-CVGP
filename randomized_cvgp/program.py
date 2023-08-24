"""Class for symbolic expression object or program."""

import array
from typing import List
import warnings

import numpy as np

np.set_printoptions(precision=4, linewidth=np.inf)

from sympy.parsing.sympy_parser import parse_expr
from sympy import pretty

from scipy.optimize import minimize
from scipy.optimize import basinhopping, direct, shgo, dual_annealing

from functions import PlaceholderConstant, Token
from const import make_const_optimizer
from utils import cached_property
import utils as U


class Program(object):
    """
    The executable program representing the symbolic expression.

    The program comprises unary/binary operators, constant placeholders
    (to-be-optimized), input variables, and hard-coded constants.

    Parameters
    ----------
    tokens : list of integers
        A list of integers corresponding to tokens in the library. "Dangling"
        programs are completed with repeated "x1" until the expression
        completes.

    Attributes
    ----------
    traversal : list
        List of operators (type: Function) and terminals (type: int, float, or
        str ("const")) encoding the pre-order traversal of the expression tree.

    tokens : np.ndarray (dtype: int)
        Array of integers whose values correspond to indices

    allow_change_tokens: np.ndarray (dtype: int)
        if each token allows to be changed during GP. 

    const_pos : list of int
        A list of indicies of constant placeholders along the traversal.

    num_changing_const: int
        number of changing constant.

    float_pos : list of float
        A list of indices of constants placeholders or floating-point constants
        along the traversal.

    sympy_expr : str
        The (lazily calculated) SymPy expression corresponding to the program.
        Used for pretty printing _only_.

    complexity : float
        The (lazily calcualted) complexity of the program.

    r : float
        The (lazily calculated) reward of the program.

    expr_objs: array of floats
        The objective functions done with opt_num_expr experiments during optimization.

    expr_consts: 2-d array of floats
        The optimized constant values with opt_num_expr experiments during optimization. 

    count : int
        The number of times this Program has been sampled.

    str : str
        String representation of tokens. Useful as unique identifier.
    """
    # Static variables
    task = None  # Task
    library = None  # Library
    const_optimizer = None  # Function to optimize constants
    cache = {}
    n_objects = 1  # Number of executable objects per Program instance

    opt_num_expr = 32  # number of experiments done for optimization

    expr_obj_thres = 1e-2  # expression objective threshold
    expr_consts_thres = 1e-3

    # Cython-related static variables
    have_cython = None  # Do we have cython installed
    execute = None  # Link to execute. Either cython or python
    cyfunc = None  # Link to cyfunc lib since we do an include inline

    noise_std = 0.0

    def __init__(self, tokens=None, allow_change_tokens=None, optimizer="BFGS"):
        """
        Builds the Program from a list of integers corresponding to Tokens.
        """

        # Can be empty if we are unpickling 
        if tokens is not None:
            self._init(tokens, allow_change_tokens)
        self.freezed = False
        self.optimizer = optimizer

    def set_vf(self, vf: list):
        """set of free variables"""
        self.vf = vf

    def pick_new_random_vf(self, verbose=False):
        """add a new variable to vf (set of free variables)"""
        candidates = [i for i in range(self.n_var) if i not in self.vf]
        if len(candidates) == 0:
            return False
        new_variable = np.random.choice(candidates)
        self.vf.append(new_variable)
        if verbose:
            print(f"{self.__repr__()}")
        return True

    def set_self_vf(self):
        appeared_variables = set([t.input_var for t in self.traversal if t.input_var is not None])
        if self.vf is None:
            self.vf = []
        for v in set(appeared_variables):
            if v not in self.vf:
                self.vf.append(v)

    def update_with_other_vf(self, other_vf, verbose=False):
        """update the set of free variables to be: the original"""
        # if np.random.rand() < 0.5 and len(other_vf) > 0:
        # print("crossover update with other vf self.vf {} other.vf {}".format(self.vf, other_vf), end='\t')
        for x in other_vf:
            if x not in self.vf:
                self.vf.append(x)

        if verbose:
            print("->".join([str(v) for v in self.vf]))

    def _init(self, tokens: np.ndarray, allow_change_tokens: np.ndarray):
        # pre-order of the program. the most important thing.
        self.traversal = [Program.library[t] for t in tokens]
        # added part: which token is allowed to be token. 1 means allowed
        self.allow_change_tokens = allow_change_tokens
        # position of the constant
        self.const_pos = [i for i, t in enumerate(self.traversal) if isinstance(t, PlaceholderConstant)]
        self.num_changing_consts = 0
        for pos in self.const_pos:  # compute num_changing_consts
            if self.allow_change_tokens[pos]:
                self.num_changing_consts += 1
        self.len_traversal = len(self.traversal)

        if self.have_cython and self.len_traversal > 1:
            self.is_input_var = [t.input_var is not None for t in self.traversal]

        self.invalid = False  # always false.
        self.tokens = tokens
        self.n_var = self.task.n_input
        self.vf = None

    def clone(self):
        new_me = Program(self.tokens, self.allow_change_tokens)

        for i in range(len(self.traversal)):
            if isinstance(self.traversal[i], PlaceholderConstant):
                new_me.traversal[i] = PlaceholderConstant(self.traversal[i].value)

        new_me.allow_change_tokens = np.copy(self.allow_change_tokens)
        new_me.tokens = np.copy(self.tokens)
        new_me.vf = self.vf

        if 'r' in self.__dict__:
            new_me.r = self.r
        if 'expr_objs' in self.__dict__:
            new_me.expr_objs = np.copy(self.expr_objs)
        if 'expr_consts' in self.__dict__:
            new_me.expr_consts = np.copy(self.expr_consts)

        return new_me

    def __getstate__(self, verbse=False):
        # for printing purpose
        have_r = "r" in self.__dict__
        have_evaluate = "evaluate" in self.__dict__
        possible_const = have_r or have_evaluate

        if verbse:
            state_dict = {
                'tokens': self.tokens.tolist(),  # string rep comes out different if we cast to array, so we can get cache misses.
                'allow_change_tokens': self.allow_change_tokens.tolist(),
                'have_r': bool(have_r),
                'r': float(self.r) if have_r else 'No r',
                'fixed_column': self.task.fixed_column,
                'have_evaluate': bool(have_evaluate),
                'evaluate': self.evaluate if have_evaluate else float(-np.inf),
                'const': array.array('d', self.get_constants()) if possible_const else float(-np.inf),
                'invalid': bool(self.invalid),
                'error_node': array.array('u', "" if not self.invalid else self.error_node),
                'error_type': array.array('u', "" if not self.invalid else self.error_type)
            }
        else:
            state_dict = {
                'vf': self.vf if self.vf is not None else 'None',
                'r': float(self.r) if have_r else 'No r',
                'tokens': self.tokens.tolist(),  # string rep comes out different if we cast to array, so we can get cache misses.
                'allow_change_tokens': self.allow_change_tokens.tolist(),
            }

        return state_dict

    def allow_change_pos(self):
        # the place the token can be changed
        return [i for i, t in enumerate(self.allow_change_tokens) if t == 1]

    def summary_constant_pos(self):
        """ return the index of 'summary constants' """
        return [pos for i, pos in enumerate(self.const_pos) if self.allow_change_tokens[pos]]

    def all_tokens_pos(self):
        # the place of the tokens that can be changed
        return [i for i in range(len(self.allow_change_tokens))]

    def subtree_end(self, subtree_start):
        # subtree_start arbitraty
        # the END point of that subtree in preorder
        k = subtree_start
        s = self.traversal[k].arity
        k += 1
        while k < self.len_traversal and s > 0:
            s -= 1 - self.traversal[k].arity
            k += 1
        return k

    def remove_r_evaluate(self):
        # remove  r
        if 'r' in self.__dict__:
            del self.__dict__['r']
        if 'evaluate' in self.__dict__:
            del self.__dict__['evaluate']
        if 'expr_objs' in self.__dict__:
            del self.__dict__['expr_objs']
        if 'expr_consts' in self.__dict__:
            del self.__dict__['expr_consts']

    def execute(self, X):
        """
        Execute program on input X.
        Parameters
        ==========
        X : np.array
            Input to execute the Program over.
        Returns
        =======
        result : np.array or list of np.array
            In a single-object Program, returns just an array. In a multi-object Program, returns a list of arrays.
        """
        if not Program.protected:
            # return some weired error.
            result, self.invalid, self.error_node, self.error_type = Program.execute_function(self.traversal, X)
        else:
            result = Program.execute_function(self.traversal, X)
            # always protected. 1/div
        return result

    def optimize(self):
        """
        Optimizes PlaceholderConstant tokens against the reward function. The
        optimized values are stored in the traversal.
        """
        # find the best constant value and fit the equation.
        if len(self.const_pos) == 0 or self.num_changing_consts == 0:
            # there is no constant in the expression
            return

        # Define the objective function: negative reward
        def f(consts):
            # replace all the constant in self.traversal with the given constant.
            self.set_constants(consts)

            # evaluate the different between predicted y and the ground truth y
            r = self.task.reward_function(self)
            # minimize the objective function
            obj = -r  # Constant optimizer minimizes the objective function

            # Need to reset to False so that a single invalid call during
            # constant optimization doesn't render the whole Program invalid.
            self.invalid = False

            return obj

        optimized_constants = []
        optimized_obj = []

        # do more than one experiment, so that we can set x2-x4 with different constant value.
        self.task.rand_draw_X_fixed()
        for expr in range(self.opt_num_expr):
            # Do the optimization
            x0 = np.random.rand(self.num_changing_consts) * 10  # Initial guess

            self.task.rand_draw_data_with_X_fixed()
            # the returned constant, and the objective function.
            # t_optimized_constants, t_optimized_obj = Program.const_optimizer(f, x0)

            if self.optimizer == "basinhopping":
                minimizer_kwargs = {"method": "Nelder-Mead",
                                    "options": {'xatol': 1e-30, 'fatol': 1e-30, 'maxiter': 1000}}
                opt_result = basinhopping(f, x0, minimizer_kwargs=minimizer_kwargs, niter=500)

                # print(opt_result)
            elif self.optimizer == 'dual_annealing':
                minimizer_kwargs = {"method": "Nelder-Mead",
                                    "options": {'xatol': 1e-30, 'fatol': 1e-30, 'maxiter': 1000}}
                lw = [-10] * self.n_var
                up = [10] * self.n_var
                bounds = list(zip(lw, up))
                opt_result = dual_annealing(f, bounds, minimizer_kwargs=minimizer_kwargs, niter=500)
            elif self.optimizer == 'shgo':

                minimizer_kwargs = {"method": "Nelder-Mead",
                                    "options": {'xatol': 1e-30, 'fatol': 1e-30, 'maxiter': 1000}}
                lw = [-10] * self.n_var
                up = [10] * self.n_var
                bounds = list(zip(lw, up))
                opt_result = shgo(f, bounds, minimizer_kwargs=minimizer_kwargs, options={'maxiter': 5})
            elif self.optimizer == "direct":
                lw = [-10] * self.n_var
                up = [10] * self.n_var
                bounds = list(zip(lw, up))
                opt_result = direct(f, bounds, maxiter=500)
            elif self.optimizer == 'Nelder-Mead':
                opt_result = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-30, 'fatol': 1e-30, 'maxiter': 1000})
            elif self.optimizer == 'CG':
                opt_result = minimize(f, x0, method='CG', options={'maxiter': 1000})
            elif self.optimizer == 'L-BFGS-B':
                opt_result = minimize(f, x0, method='L-BFGS-B', options={'maxiter': 1000})
            elif Program.noise_std > 0:
                opt_result = minimize(f, x0, method='Nelder-Mead', options={'eps': Program.noise_std})
            else:
                # change the method from BFGS to Nelder-Mead to improve the precision.
                # opt_result = minimize(f, x0, method='Nelder-Mead', options={'xatol': 1e-30, 'fatol': 1e-30, 'maxiter': 1000})
                opt_result = minimize(f, x0, method='BFGS', options={'maxiter': 1000})

            t_optimized_constants = opt_result['x']
            t_optimized_obj = opt_result['fun']

            optimized_constants.append(t_optimized_constants)

            # add validated data as the obj
            self.task.rand_draw_data_with_X_fixed()
            validate_obj = -self.task.reward_function(self)
            optimized_obj.append(validate_obj)

        optimized_obj = np.array(optimized_obj)
        optimized_constants = np.array(optimized_constants)

        # print('optimized_obj=', optimized_obj)
        # print('optimized_consts=', optimized_constants)
        # if obj close to zero, we get a good expression.
        # rember all the objective function and constant across all the iterations, so that we know which one is a real constant, which one is a variable.
        self.expr_objs = optimized_obj
        self.expr_consts = optimized_constants

        assert self.expr_objs.shape[0] == self.opt_num_expr
        assert len(self.expr_objs.shape) == 1

        # print('expr_objs=', self.expr_objs.tolist())
        # print('expr_consts=', self.expr_consts.tolist())

        # Set the optimized constants
        # set the value of optimized constants with the last optimized constants
        # (the values of the constants may change, so only the last one makes sense; the mean does not make sense). Nan Comments: Why not use average?
        self.set_constants(t_optimized_constants)

    def freeze_equation(self):
        if len(self.const_pos) == 0 or self.num_changing_consts == 0:
            assert "r" in self.__dict__, 'reward is not included'
            if self.r >= -self.expr_obj_thres:
                for pos, t in enumerate(self.traversal):
                    self.allow_change_tokens[pos] = 0
            print("freeze_equation->allow_change_tokens: {}".format(self.allow_change_tokens))
            return

        assert 'expr_objs' in self.__dict__
        # fitted objective  <= thereshold (residual is 0.01)

        if np.max(self.expr_objs) <= self.expr_obj_thres:
            print("objective residual: {}, threshold {}".format(self.expr_objs, self.expr_obj_thres))
            self.freezed = True
            # print(new_program, self.traversal == new_program)
            for pos, t in enumerate(self.traversal):
                if not isinstance(t, PlaceholderConstant):
                    # residual is within threshold and is not a constant, freeze it.
                    self.allow_change_tokens[pos] = 0

            # compute num_changing_consts and set for the constants
            for i, pos in enumerate(self.const_pos):
                print("constant std: {}, threshold {}".format(np.std(self.expr_consts[:, i]), self.expr_consts_thres))
                if np.std(self.expr_consts[:, i]) <= self.expr_consts_thres:
                    self.allow_change_tokens[pos] = 0
                self.num_changing_consts += self.allow_change_tokens[pos]

    def get_constants(self):
        """Returns the values of a Program's constants."""
        return [t.value for t in self.traversal if isinstance(t, PlaceholderConstant)]

    def set_constants(self, consts):
        """Sets the program's constants to the given values"""
        consts_tp = 0
        for i, pos in enumerate(self.const_pos):
            # Create a new instance of PlaceholderConstant instead of changing
            # the "values" attribute, otherwise all Programs will have the same
            # instance and just overwrite each other's value.
            if self.allow_change_tokens[pos]:
                assert U.is_float(consts[consts_tp]), "Input to program constants must be of a floating point type"
                self.traversal[pos] = PlaceholderConstant(consts[consts_tp])
                consts_tp += 1

    @classmethod
    def clear_cache(cls):
        """Clears the class' cache"""
        cls.cache = {}

    @classmethod
    def set_task(cls, task):
        """Sets the class' Task"""
        Program.task = task
        Program.library = task.library

    @classmethod
    def set_const_optimizer(cls, name, **kwargs):
        """Sets the class' constant optimizer"""
        const_optimizer = make_const_optimizer(name, **kwargs)
        Program.const_optimizer = const_optimizer

    @classmethod
    def set_complexity(cls, name):
        """Sets the class' complexity function"""

        all_functions = {
            # No complexity
            None: lambda p: 0.0,
            # Length of sequence
            "length": lambda p: len(p.traversal),
            # Sum of token-wise complexities
            "token": lambda p: sum([t.complexity for t in p.traversal]),
        }

        assert name in all_functions, "Unrecognzied complexity function name."

        Program.complexity_function = lambda p: all_functions[name](p)

    @classmethod
    def set_execute(cls, protected):
        """Sets which execute method to use"""

        # Check if cython_execute can be imported; if not, fall back to python_execute
        # try:
        import cyfunc
        from execute import cython_execute
        execute_function = cython_execute
        Program.have_cython = True
        # except ImportError:
        #     from execute import python_execute
        #     execute_function = python_execute
        #     Program.have_cython = False

        if protected:
            Program.protected = True
            Program.execute_function = execute_function
        else:
            Program.protected = False

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

            Program.execute_function = unsafe_execute

    def _set_dataX_allowed_input_tokens(self, allowed_input_token, verbose=False):
        """Input is a set of free input variables"""
        free_input_tokens = np.zeros(self.n_var, dtype=np.int32)
        if allowed_input_token:
            for vari in allowed_input_token:
                free_input_tokens[vari] = 1
        self.task.set_allowed_inputs(free_input_tokens)
        if verbose:
            print("For dataX: {} Program.task.allowed_input:{} fixed_column:{}".format(
                allowed_input_token, Program.task.allowed_input, Program.task.fixed_column))

    @cached_property
    def r(self):
        """Evaluates and returns the reward of the program"""
        self._set_dataX_allowed_input_tokens(self.vf, verbose=False)
        with warnings.catch_warnings():
            # print('===before optimize===')

            # Optimize any PlaceholderConstants
            self.optimize()

            # print('===after optimize===')

            # XYX: may need to average over multiple runs. 
            # Return final reward as the mean of multiple experiment runs.
            # Using another reward_function() evaluation seems not make sense.
            if 'expr_objs' in self.__dict__:
                return -np.mean(self.expr_objs)
            else:
                # this means there is no constants to be optimized.
                self.expr_objs = []
                self.task.rand_draw_X_fixed()
                # Nan: note that the values of controled variable stay the same for `opt_num_expr` tryouts.
                for expr in range(self.opt_num_expr):
                    self.task.rand_draw_data_with_X_fixed()
                    self.expr_objs.append(self.task.reward_function(self))
                self.expr_objs = np.array(self.expr_objs)
                return np.mean(self.expr_objs)

    @cached_property
    def evaluate(self):
        """Evaluates and returns the evaluation metrics of the program."""

        # Program must be optimized before computing evaluate
        if "r" not in self.__dict__:
            print("WARNING: Evaluating Program before computing its reward. Program will be optimized first.")
            self.optimize()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            return self.task.evaluate(self)

    @cached_property
    def sympy_expr(self):
        """
        Returns the attribute self.sympy_expr.

        This is actually a bit complicated because we have to go: traversal -->
        tree --> serialized tree --> SymPy expression
        """
        tree = self.traversal.copy()
        tree = build_tree(tree)
        tree = convert_to_sympy(tree)
        try:
            expr = parse_expr(tree.__repr__())  # SymPy expression
        except:
            expr = tree.__repr__()
        return expr

    def pretty(self):
        """Returns pretty printed string of the program"""
        return pretty(self.sympy_expr)

    def print_expression(self):
        print("{}".format(self.traversal))

    def __repr__(self):
        """Prints the program's traversal"""
        return ','.join([repr(t) for t in self.traversal])


###############################################################################
# Everything below this line is currently only being used for pretty printing #
###############################################################################


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):
        self.val = val
        self.children = []

    def __repr__(self):
        children_repr = ",".join(repr(child) for child in self.children)
        if len(self.children) == 0:
            return self.val  # Avoids unnecessary parantheses, e.g. x1()
        return "{}({})".format(self.val, children_repr)


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    n_children = op.arity
    val = repr(op)
    if val in capital:
        val = val.capitalize()

    node = Node(val)

    for _ in range(n_children):
        node.children.append(build_tree(traversal))

    return node


# this function is used for pretty print the expression
def convert_to_sympy(node):
    """Adjusts trees to only use node values supported by sympy"""

    if node.val == "div":
        node.val = "Mul"
        new_right = Node("Pow")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "sub":
        node.val = "Add"
        new_right = Node("Mul")
        new_right.children.append(node.children[1])
        new_right.children.append(Node("-1"))
        node.children[1] = new_right

    elif node.val == "inv":
        node.val = Node("Pow")
        node.children.append(Node("-1"))

    elif node.val == "neg":
        node.val = Node("Mul")
        node.children.append(Node("-1"))

    elif node.val == "n2":
        node.val = "Pow"
        node.children.append(Node("2"))

    elif node.val == "n3":
        node.val = "Pow"
        node.children.append(Node("3"))

    elif node.val == "n4":
        node.val = "Pow"
        node.children.append(Node("4"))

    for child in node.children:
        convert_to_sympy(child)

    return node
