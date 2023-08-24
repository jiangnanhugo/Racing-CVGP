"""Utility functions used in deep symbolic optimization."""

import collections
import copy
import functools
import numpy as np
import time
import itertools


# def unique(arr):
#     return list(set(arr))


class Tree(object):
    """
    this data structure maintain different variable ordering as a tree.
    """

    def __init__(self, nvar, max_width=6):
        self.layers = nvar + 2
        self.maxwidth_pool_idxes = [[] for i in range(self.layers)]
        self.max_width = max_width
        self.maxwidth_pool_idxes[-1].append(Node((-1,), (-1,), (-1,)))
        self.maxwidth_pool_idxes[0].append(Node((-1,), (-1,), (-1,)))

    def insert_node(self, one_node):
        if not one_node:
            return False
        if one_node in self.maxwidth_pool_idxes[len(one_node.cur)]:
            print("new_pool_idx {} already discovered {}".format(one_node, self.maxwidth_pool_idxes[len(one_node.cur)]))
            return False

        if self.max_width > 0 and len(self.maxwidth_pool_idxes[len(one_node.cur)]) > self.max_width:
            print(f"max width reached {len(one_node.cur)}-layer {len(self.maxwidth_pool_idxes[len(one_node.cur)])}")
            return False

        self.maxwidth_pool_idxes[len(one_node.cur)].append(one_node)
        return True

    def all_pair_combinations(self):
        visited = set()
        historical_pools_idxes = []
        for i in range(self.layers):
            for node in self.maxwidth_pool_idxes[i]:
                if node.cur not in visited:
                    historical_pools_idxes.append(node)
                    visited.add(node.cur)
        to_be_merged_pool_pairs = []
        if len(historical_pools_idxes) != 0:
            for one_pool_idx, another_pool_idx in itertools.combinations(historical_pools_idxes, r=2):
                to_be_merged_pool_pairs.append((one_pool_idx, another_pool_idx))
        return to_be_merged_pool_pairs

    def combine_with_one_var_pool_idxes(self, given_layer, chosen_layer=1):
        one_var_pool_indexes = []
        if given_layer + 2 == self.layers:
            chosen_layer = 0
        for node in self.maxwidth_pool_idxes[chosen_layer]:
            one_var_pool_indexes.append(node)
        #
        historical_pools_idxes = []

        for node in self.maxwidth_pool_idxes[given_layer]:
            if node not in historical_pools_idxes:
                historical_pools_idxes.append(node)
        to_be_merged_pool_pairs = []
        if len(historical_pools_idxes) != 0:
            for one_pool_idx, another_pool_idx in itertools.product(historical_pools_idxes, one_var_pool_indexes):
                to_be_merged_pool_pairs.append((one_pool_idx, another_pool_idx))
        if given_layer + 2 == self.layers:
            to_be_merged_pool_pairs = [(y, x) for x, y in to_be_merged_pool_pairs]
        return to_be_merged_pool_pairs


# this node is used for keep track of variable ordering
class Node(object):
    def __init__(self, l: tuple = None, r: tuple = None, cur: tuple = None):
        # l:left parent pool idx. It is a tuple or None, r right parent pool idx
        self.l = l
        self.r = r
        if not cur:
            new_pool_idx = self.l + self.r
            self.cur = tuple(new_pool_idx)
        else:
            self.cur = cur

    def __eq__(self, other):
        if not other:
            return False
        if self.l == other.l and self.r == other.r and self.cur == other.cur:
            return True
        return False

    def __repr__(self):
        return f"{self.l},{self.r}->{self.cur}"

    def __hash__(self):
        return hash(f"{self.l},{self.r}->{self.cur}")


def create_node(one_pool_idx, another_pool_idx):
    # assert len(another_pool_idx.cur) == 1, "another pool must be one variable!"
    if one_pool_idx == another_pool_idx or another_pool_idx.cur[0] in set(one_pool_idx.cur):
        return None
    new_pool_idx = one_pool_idx.cur + another_pool_idx.cur
    new_pool_idx = tuple(new_pool_idx)
    if new_pool_idx == one_pool_idx.cur or new_pool_idx == another_pool_idx.cur:
        return None
    return Node(one_pool_idx.cur, another_pool_idx.cur, new_pool_idx)


def create_geometric_generations(n_generations, nvar):
    gens = [0] * nvar
    for it in range(nvar - 1, 0, -1):
        gens[it] = int(n_generations // 3)
        n_generations -= gens[it]
    gens[0] = n_generations
    # for it in range(0, nvar):
    #     if gens[it] < 50:
    #         gens[it] = 50
    # print('generation #:', gens, 'sum=', sum(gens))
    return gens[::-1]


def create_uniform_generations(n_generations, nvar):
    gens = [0] * nvar
    each_gen = n_generations // nvar
    for it in range(nvar - 1, 0, -1):
        gens[it] = each_gen
        n_generations -= each_gen
    gens[0] = n_generations
    print('generation #:', gens, 'sum=', sum(gens))
    return gens


def is_float(s):
    """Determine whether the input variable can be cast to float."""
    try:
        float(s)
        return True
    except ValueError:
        return False


# Adapted from: https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
def is_pareto_efficient(costs):
    """
    Find the pareto-efficient points given an array of costs.

    Parameters
    ----------

    costs : np.ndarray
        Array of shape (n_points, n_costs).

    Returns
    -------

    is_efficient_maek : np.ndarray (dtype:bool)
        Array of which elements in costs are pareto-efficient.
    """

    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index < len(costs):
        nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
    is_efficient_mask = np.zeros(n_points, dtype=bool)
    is_efficient_mask[is_efficient] = True
    return is_efficient_mask


class cached_property(object):
    """
    Decorator used for lazy evaluation of an object attribute. The property
    should be non-mutable, since it replaces itself.
    """

    def __init__(self, getter):
        self.getter = getter

        functools.update_wrapper(self, getter)

    def __get__(self, obj, cls):
        if obj is None:
            return self

        value = self.getter(obj)
        setattr(obj, self.getter.__name__, value)
        return value


def weighted_quantile(values, weights, q):
    """
    Computes the weighted quantile, equivalent to the exact quantile of the
    empirical distribution.

    Given ordered samples x_1 <= ... <= x_n, with corresponding weights w_1,
    ..., w_n, where sum_i(w_i) = 1.0, the weighted quantile is the minimum x_i
    for which the cumulative sum up to x_i is greater than or equal to 1.

    Quantile = min{ x_i | x_1 + ... + x_i >= q }
    """

    sorted_indices = np.argsort(values)
    sorted_weights = weights[sorted_indices]
    sorted_values = values[sorted_indices]
    cum_sorted_weights = np.cumsum(sorted_weights)
    i_quantile = np.argmax(cum_sorted_weights >= q)
    quantile = sorted_values[i_quantile]

    # NOTE: This implementation is equivalent to (but much faster than) the
    # following:
    # from scipy import stats
    # empirical_dist = stats.rv_discrete(name='empirical_dist', values=(values, weights))
    # quantile = empirical_dist.ppf(q)

    return quantile


# Entropy computation in batch
def empirical_entropy(labels):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log(i)

    return ent


def get_duration(start_time):
    return get_human_readable_time(time.time() - start_time)


def get_human_readable_time(s):
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    return "{:02d}:{:02d}:{:02d}:{:05.2f}".format(int(d), int(h), int(m), s)


def safe_merge_dicts(base_dict, update_dict):
    """Merges two dictionaries without changing the source dictionaries.

    Parameters
    ----------
        base_dict : dict
            Source dictionary with initial values.
        update_dict : dict
            Dictionary with changed values to update the base dictionary.

    Returns
    -------
        new_dict : dict
            Dictionary containing values from the merged dictionaries.
    """
    if base_dict is None:
        return update_dict
    base_dict = copy.deepcopy(base_dict)
    for key, value in update_dict.items():
        if isinstance(value, collections.Mapping):
            base_dict[key] = safe_merge_dicts(base_dict.get(key, {}), value)
        else:
            base_dict[key] = value
    return base_dict
