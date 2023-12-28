from sympy import Symbol
import json


class KnownEquation(object):
    _eq_name = None
    _function_set = ['add', 'sub', 'mul', 'div', 'inv', 'pow', 'sin', 'cos', 'exp', 'log', 'const']

    def __init__(self, num_vars, vars_range_and_types=None, kwargs_list=None):
        if kwargs_list is None:
            kwargs_list = [{'real': True} for _ in range(num_vars)]

        assert len(kwargs_list) == num_vars
        self.num_vars = num_vars
        self.vars_range_and_types = vars_range_and_types
        self.x = [Symbol(f'X_{i}', **kwargs) for i, kwargs in enumerate(kwargs_list)]
        self.sympy_eq = None

    def vars_range_and_types_to_json_str(self):
        if self.vars_range_and_types:
            return json.dumps([one.to_dict() for one in self.vars_range_and_types])
        else:
            default = {'name': 'LogUniform',
                       'range': [0.1, 10],
                       'only_positive': True}
            return json.dumps([default for _ in range(self.num_vars)])


class DefaultSampling(object):
    def __init__(self, name, min_value, max_value, only_positive=False):
        self.name = name
        self.range = [min_value, max_value]
        self.only_positive = only_positive

    def to_dict(self):
        return {'name': self.name,
                'range': self.range,
                'only_positive': self.only_positive}


class LogUniformSampling(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False):
        super().__init__('LogUniform', min_value, max_value, only_positive)


class IntegerUniformSampling(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False):
        super().__init__('IntegerUniform', int(min_value), int(max_value), only_positive)


class UniformSampling(DefaultSampling):
    def __init__(self, min_value, max_value, only_positive=False):
        super().__init__('Uniform', min_value, max_value, only_positive)
