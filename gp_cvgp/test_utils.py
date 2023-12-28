import functions
from library import Library, Token, PlaceholderConstant
import execute

from program import Program
import regress_task
from const import ScipyMinimize

import gp_xyx

import numpy as np

def test_optimize():
    nvar = 5
    regress_batchsize = 255
    opt_num_expr = 1
    
    expr_obj_thres = 1e-2
    expr_consts_thres = 1e-3
    
    
    # get all the functions and variables ready
    var_x = []
    for i in range(nvar):
        xi = Token(None, 'X_'+str(i), 0, 0., i)
        var_x.append(xi)

    ops = [
        # Binary operators
        Token(np.add, "add", arity=2, complexity=1),
        Token(np.subtract, "sub", arity=2, complexity=1),
        Token(np.multiply, "mul", arity=2, complexity=1)
        #functions.protected_ops[0],  # 'div'
        #functions.protected_ops[5]   # 'inv' '1/x'
    ]
    named_const = [PlaceholderConstant(1.0)]
    protected_library = Library(ops + var_x + named_const)

    protected_library.print_library()

    # get program ready
    Program.library = protected_library
    Program.opt_num_expr = opt_num_expr
    Program.expr_obj_thres = expr_obj_thres
    Program.expr_consts_thres = expr_consts_thres
    
    Program.set_execute(True) #protected = True

    # set const_optimizer
    Program.const_optimizer = ScipyMinimize()

    # create the true program
    # x_0 + 3 x_2 + 5 x_4 - 6 x_2 x_4 + 9 x_0 x_2
    preorder = ['add', \
                      'add', 'add', 'X_0', 'mul', 'const', 'X_2', \
                             'mul', 'const', 'X_4', \
                      'sub', 'mul', 'const', 'mul', 'X_0', 'X_2', \
                             'mul', 'const', 'mul', 'X_2', 'X_4']
    preorder_actions = protected_library.actionize(preorder)
    true_pr_allow_change = np.zeros(len(preorder), dtype=np.int32)
    true_pr = Program(preorder_actions, true_pr_allow_change)
    
    true_pr.traversal[5] = PlaceholderConstant(3.0) 
    true_pr.traversal[8] = PlaceholderConstant(5.0) 
    true_pr.traversal[12] = PlaceholderConstant(9.0) 
    true_pr.traversal[17] = PlaceholderConstant(6.0)

    # # x_0 + 3 x_2
    # preorder = ['add', 'X_0', 'mul', 'const', 'X_2']
    # preorder_actions = protected_library.actionize(preorder)
    # true_pr_allow_change = np.zeros(len(preorder), dtype=np.int32)
    # true_pr = Program(preorder_actions, true_pr_allow_change)
    
    # true_pr.traversal[3] = PlaceholderConstant(3.0) 
   
    # x_0 + 3
    # preorder = ['add', 'X_0', 'const']
    # preorder_actions = protected_library.actionize(preorder)
    # true_pr_allow_change = np.zeros(len(preorder), dtype=np.int32)
    # true_pr = Program(preorder_actions, true_pr_allow_change)
    
    # true_pr.traversal[2] = PlaceholderConstant(3.0) 
 
    # set the task
    allowed_input_tokens = np.zeros(nvar, dtype=np.int32) 
    allowed_input_tokens[0] = 1
    Program.task = regress_task.RegressTaskV1(regress_batchsize,
                                              allowed_input_tokens, true_pr)


    # set test program
    preorder = ['add', 'const', 'mul', 'X_0', 'const']
    preorder_actions = protected_library.actionize(preorder)
    test_pr_allow_change = np.ones(len(preorder), dtype=np.int32)
    test_pr = Program(preorder_actions, test_pr_allow_change)
    
    test_pr.optimize()

    print('test_pr=', test_pr.__getstate__())
    
    bb = 9*Program.task.X_fixed[2] + 1
    cc = 3*Program.task.X_fixed[2] + 5*Program.task.X_fixed[4] - 6*Program.task.X_fixed[2]*Program.task.X_fixed[4]
    manual = np.array([cc, bb])
    test_pr.set_constants(manual)
    print('check true param r=', Program.task.reward_function(test_pr))
    
   


if __name__ == '__main__':
    test_optimize()
