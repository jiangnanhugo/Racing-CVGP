import pickle
import os
from sympy import *
import click
from sympy.parsing.sympy_parser import parse_expr


def read_true_program(filename):
    prog = pickle.load(open(filename, 'rb'))
    preorder_traversal_expr = prog['preorder']
    preorder_traversal_tuple = []
    for loc, val in zip(prog['const_loc'], prog['consts']):
        preorder_traversal_expr[loc] = val
    for idx, it in enumerate(preorder_traversal_expr):
        if idx in prog['const_loc']:
            preorder_traversal_tuple.append((it, 'const'))
        elif it.startswith('x') or it.startswith('X'):
            preorder_traversal_tuple.append((it, 'var'))
        elif it in ['add', 'mul', 'sub', 'div']:
            preorder_traversal_tuple.append((it, 'binary'))
        elif it in ['inv', 'sqrt', 'sin', 'cos', 'exp', 'log', 'n2', 'n3', 'n4']:
            preorder_traversal_tuple.append((it, 'unary'))
    return preorder_traversal_expr, preorder_traversal_tuple


#
def sympy_expr(traversal):
    """
    Returns the attribute self.sympy_expr.

    This is actually a bit complicated because we have to go:
    traversal --> tree --> serialized tree --> SymPy expression
    """
    tree = build_tree(traversal)
    tree = convert_to_sympy(tree)
    tree_str = tree.__repr__()
    expr = parse_expr(tree_str)
    return expr


# Possible library elements that sympy capitalizes
capital = ["add", "mul", "pow"]


class Node(object):
    """Basic tree class supporting printing"""

    def __init__(self, val):

        self.val = val
        self.children = []

    def __repr__(self):
        if len(self.children) == 0:
            return "{}".format(self.val)

        if len(self.children) == 0:
            return self.val
        if self.val == "add":
            return "{} + {}".format(repr(self.children[0]), repr(self.children[1]))
        elif self.val == "mul":
            return "{} * {}".format(repr(self.children[0]), repr(self.children[1]))
        elif self.val == "inv":
            children_repr = ",".join(repr(child) for child in self.children)
            return "1 / {}".format(children_repr)
        else:
            children_repr = ",".join(repr(child) for child in self.children)
            return "{}({})".format(self.val, children_repr)


op_arity_dict = {
    'add': 2, 'sub': 2, 'mul': 2, 'div': 2, 'inv': 1, 'sqrt': 1,
    'sin': 1, 'cos': 1, 'exp': 1, 'log': 1, 'n2': 1, 'n3': 1, 'n4': 1}


def build_tree(traversal):
    """Recursively builds tree from pre-order traversal"""

    op = traversal.pop(0)
    if op in op_arity_dict:
        n_children = op_arity_dict[op]
    else:
        node = Node(op)
        return node
    val = op
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


template = """@register_eq_class
class {}(KnownEquation):
    _eq_name = '{}'
    _function_set = {}
    
    def __init__(self):
        super().__init__(num_vars={})
        x = self.x
        self.sympy_eq = {}
        self.sympy_eq_preorder_traversal = {}
"""


def write_to_files(equations, template, output_folder):
    fw = open(os.path.join(output_folder, "equations_trigometric_extra.py"), 'w')

    for spl in equations:
        function_set = None
        for key in function_set_dict:
            if key in spl[0]:
                function_set = function_set_dict[key]
        fw.write(template.format(
            spl[0].replace("-", "_"),
            spl[0],
            function_set,
            spl[1],
            spl[2].replace('X_0', 'x[0]').replace('X_1', 'x[1]').replace('X_2', 'x[2]').replace('X_3', 'x[3]').replace(
                'X_4', 'x[4]').replace('X_5', 'x[5]').replace('X_6', 'x[6]').replace('X_7', 'x[7]').replace('X_8', 'x[8]').replace('X_9',
                                                                                                                                   'x[9]').replace(
                'X_10', 'x[10]').replace('X_11', 'x[11]').replace(
                'log', 'sympy.log').replace('exp', 'sympy.exp').replace('sin', 'sympy.sin').replace(
                'cos', 'sympy.cos').replace('div', 'sympy.div').replace('sqrt', 'sympy.sqrt').replace('pow', 'sympy.pow'),
            spl[3])
        )


function_set_dict = {
    'inv_': ["add", "sub", "mul", "div", "inv", "const"],
    'sincos_': ["add", "sub", "mul", "sin", "cos", "const"],
    'sincosinv_': ["add", "sub", "mul", "div", "inv", "sin", "cos", "const"],
}


@click.command()
@click.option('--basepath', default='/home/jiangnan/PycharmProjects/xyx_dso/data/')
@click.option('--output_folder', default='./')
def main(basepath, output_folder):
    program_files = []
    for root, dirs, files in os.walk(basepath, topdown=False):
        for name in files:
            if name.endswith(".data"):
                program_files.append(os.path.join(root, name))
    equations = []
    for filename in program_files:
        preorder_traversal_expr, preorder_traversal_tuple = read_true_program(filename)
        expr = sympy_expr(preorder_traversal_expr)
        num_vars = len(expr.free_symbols)
        eq_class_name = "_".join(filename.split("/")[-2:])[:-5]
        equations.append([eq_class_name, num_vars, str(expr), preorder_traversal_tuple])

    write_to_files(equations, template, output_folder)


if __name__ == '__main__':
    main()
