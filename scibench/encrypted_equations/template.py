from equations_feynman import *

if __name__ == '__main__':
    var_eq_dict = {}

    for key in EQUATION_CLASS_DICT:
        nvar = EQUATION_CLASS_DICT[key]().num_vars
        if nvar not in var_eq_dict:
            var_eq_dict[nvar] = [EQUATION_CLASS_DICT[key]]
        else:
            var_eq_dict[nvar].append(EQUATION_CLASS_DICT[key])
    for i in range(1,9):
        print(f"var{i}=", end="[")
        for ob in var_eq_dict[i]:
            name = ob().__class__.__name__
            print("'"+name +"'", end=",  ")
        print(']')

