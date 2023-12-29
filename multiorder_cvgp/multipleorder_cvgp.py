import time
import numpy as np
from operator import attrgetter
from program import Program
from utils import Node, Tree, create_node, create_uniform_generations, create_geometric_generations



class ExpandingGeneticProgram(object):
    """
    populations: the current list of programs
    hofs: list of the best programs
    """

    # static variables
    library = None
    gp_helper = None

    def __init__(self, cxpb, mutpb, maxdepth, population_size, tour_size, hof_size, n_generations, nvar):
        """
        cxpb: probability of mate
        mutpb: probability of mutations
        maxdepth: the maxdepth of the tree during mutation
        population_size: the size of the selected populations (at the end of each generation)
        tour_size: the size of the tournament for selection
        hof_size: the size of the best programs retained
        n_generations: the number of generations to be applied over each pool
        nvar: number of variables
        """
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.maxdepth = maxdepth
        self.population_size = population_size
        self.tour_size = tour_size
        self.hof_size = hof_size

        self.n_generations = create_uniform_generations(n_generations, nvar + 1)
        print(self.n_generations)

        self.timer_log = []
        self.gen_num = 0

        self.nvar = nvar
        assert self.library != None
        assert Program.task != None

    def create_init_population(self):
        """
           create the initial population; for every variable, a set of generate random equations.
           save them to self.populations, self.hofs.
           for every single pool: look for every token in library, fill in the leaves with constants or inputs.
        """
        self.populations = dict()
        self.hofs = dict()
        current_node_lists = []
        for vari in range(self.nvar):
            tmp_node = Node(l=-1, r=-1, cur=(vari,))
            self._set_allowed_input_tokens(tmp_node)
            self.variable_ordering_tree.insert_node(tmp_node)
            current_node_lists.append(tmp_node)
            for i, t in enumerate(self.library.tokens):
                if self.library.allowed_tokens[i]:
                    tree = [i]
                    for j in range(t.arity):
                        t_idx = np.random.choice(self.library.tokens_of_arity[0])
                        while self.library.allowed_tokens[t_idx] == 0:
                            t_idx = np.random.choice(self.library.tokens_of_arity[0])
                        tree.append(t_idx)
                    tree = np.array(tree)
                    pr = Program(tree, np.ones(tree.size, dtype=np.int32))
                    if tmp_node not in self.populations:
                        self.populations[tmp_node] = []
                    self.populations[tmp_node].append(pr)
                    new_pr = pr.clone()
                    if tmp_node not in self.hofs:
                        self.hofs[tmp_node] = []
                    self.hofs[tmp_node].append(new_pr)
        return current_node_lists

    def run_with_multiple_randomized_experiment_schedule(self, maximum_width=10):
        # 1. generate all single variable equations by creating `#nvar` POOLS.
        self.variable_ordering_tree = Tree(self.nvar, max_width=maximum_width)
        current_node_lists = self.create_init_population()
        print("=" * 20 + "Init Population" + "=" * 20)
        for cur_node in self.populations:
            for pr in self.populations[cur_node]:
                pr.print_expression()
            print('-' * 50)

        # 2. apply GP for every single POOL
        for round_idx in range(self.nvar + 1):
            if round_idx == self.nvar:
                Program.task.batchsize *= Program.opt_num_expr
                Program.opt_num_expr = 1
            for cur_node in current_node_lists:
                print(cur_node)
                # 2.1 set the free variables and controlled variables for the given POOL
                self._set_allowed_input_tokens(cur_node)
                # 2.2 re-evaluate the constants and reward for the given POOL
                for pr in self.populations[cur_node]:
                    # a cached property in python (evaluated once) force the function to evaluate a new r
                    pr.remove_r_evaluate()
                    thisr = pr.r  # goodness-of-fit
                for pr in self.hofs[cur_node]:
                    pr.remove_r_evaluate()
                    thisr = pr.r

                # 2.3 for the given POOL, do n generation of GP, find the best set of fitted expressions
                for it in range(self.n_generations[round_idx]):
                    print('++++++++++++ VAR {} ITERATION {} ++++++++++++'.format(cur_node.cur, it))
                    self.one_generation(cur_node)

                self.update_population(cur_node)
                print(f'populations {cur_node}')
                print_prs(self.populations[cur_node])

                # 2.4 freeze tokens in the expressions
                for i, pr in enumerate(self.populations[cur_node]):
                    # evaluate r again, just incase it has not been evaluated.
                    _ = pr.r
                    if len(pr.const_pos) == 0 or pr.num_changing_consts == 0:
                        # only expand at those constant node. if there are no constant node,then we are done
                        # if we do not want num_changing_consts, then we also quit.
                        print('there are no constant node. we are done...')
                    else:
                        if not ("expr_objs" in pr.__dict__ and "expr_consts" in pr.__dict__):
                            print('WARNING: pr.expr_objs NOT IN DICT: pr=' + str(pr.__getstate__()))
                            pr.remove_r_evaluate()
                            _ = pr.r
                    # whether you get very different value for different constant.
                    pr.freeze_equation()

            # pick two pools randomly, create a new pool of expression containing expression with the union of free variables
            if len(current_node_lists) == 0:
                current_node_lists = []
                continue
            print("all the pools", current_node_lists)
            ## 3. generate a lot of different pairs of pools that can be merged
            to_be_merged_pool_pairs = self.variable_ordering_tree.combine_with_one_var_pool_idxes(round_idx + 1)
            print("to be merged:", [[x.cur, y.cur] for x, y in to_be_merged_pool_pairs])
            ## 3.1 use randomized stategies to pick new pools
            np.random.shuffle(to_be_merged_pool_pairs)
            new_pool_idxes = []
            for one_pool_idx, another_pool_idx in to_be_merged_pool_pairs:
                tmp_node = create_node(one_pool_idx, another_pool_idx)
                is_success = self.variable_ordering_tree.insert_node(tmp_node)
                if not is_success:
                    continue
                print(tmp_node)
                one_joint_pool = self.create_new_pools(one_pool_idx, another_pool_idx)
                self.populations[tmp_node] = one_joint_pool
                self.hofs[tmp_node] = one_joint_pool
                new_pool_idxes.append(tmp_node)
            current_node_lists = new_pool_idxes

    def create_new_pools(self, one_pool_idx, another_pool_idx):
        """
        Given two pools of equations, pick two equations from two pools and apply m
        """
        if another_pool_idx not in self.populations or len(self.populations[another_pool_idx]) == 0:
            return self.populations[one_pool_idx]

        if one_pool_idx not in self.populations or len(self.populations[one_pool_idx]) == 0:
            return self.populations[another_pool_idx]
        return self.populations[one_pool_idx]

    def one_generation(self, pool_idx, verbose=False):
        """
        One step of the genetic algorithm.
        This wraps selection, mutation, crossover and hall of fame computation
        over all the individuals in the population for this epoch/step.
        Parameters
        ----------
        pool_idx : int. the set of equations.
        """
        t1 = time.perf_counter()

        # Select the next generation individuals
        offspring = self.selectTournament(self.population_size, self.tour_size, pool_idx)
        if verbose:
            print('offspring after select=')
            print_prs(offspring)

        # Vary the pool of individuals
        offspring = self._var_and(offspring)
        if verbose:
            print('offspring after mutation and cross-over=')
            print_prs(offspring)

        # Replace the current population by the offspring
        self.populations[pool_idx] = offspring + self.hofs[pool_idx]

        # Update hall of fame
        self.update_hof(pool_idx)
        if verbose:
            print("after update hof=")
            print_prs(self.hofs[pool_idx])
        timer = time.perf_counter() - t1
        self.timer_log.append(timer)

    def update_hof(self, pool_idx):
        """update the set of Hall of Fame for the given pool_idx pool"""
        new_hof = sorted(self.populations[pool_idx], reverse=True, key=attrgetter('r'))

        self.hofs[pool_idx] = []
        for i in range(self.hof_size):
            pr = new_hof[i]
            # if pr.r == np.nan or pr.r == np.inf or pr.r == -np.inf:
            #     print("filter:", pr.r, pr.__getstate__(), end="\t")
            #     pr.print_expression()
            #     continue
            self.hofs[pool_idx].append(pr.clone())

    def update_population(self, pool_idx):
        """update the population in the given indexed pool. sort by fitness score and cut by population_size"""
        filtered_population = []
        for pr in self.populations[pool_idx]:
            # if pr.r == np.nan or pr.r == np.inf or pr.r == -np.inf:
            #     print("filter:", pr.r, pr.__getstate__(), end="\t")
            #     pr.print_expression()
            #     continue
            filtered_population.append(pr)
        new_population = sorted(filtered_population, reverse=True, key=attrgetter('r'))
        self.populations[pool_idx] = []
        for i in range(min(self.population_size, len(filtered_population))):
            self.populations[pool_idx].append(new_population[i].clone())

    def selectTournament(self, population_size, tour_size, pool_idx):
        offspring = []
        for pp in range(population_size):
            spr = np.random.choice(self.populations[pool_idx], tour_size)
            maxspr = max(spr, key=attrgetter('r'))
            maxspri = maxspr.clone()
            offspring.append(maxspri)
        return offspring

    def _var_and(self, offspring):
        """Apply crossover AND mutation to each individual in a population, given a constant probability."""

        # Apply crossover on the offspring
        np.random.shuffle(offspring)
        for i in range(1, len(offspring), 2):
            if np.random.random() < self.cxpb:
                self.gp_helper.mate(offspring[i - 1], offspring[i])

        # Apply mutation on the offspring
        for i in range(len(offspring)):
            if np.random.random() < self.mutpb:
                self.gp_helper.multi_mutate(offspring[i], self.maxdepth)

        return offspring

    def _set_allowed_input_tokens(self, node):
        """Input is a set of free input variables"""
        allowed_input_token, library_disallowed_input_token = node.cur, node.l
        free_input_tokens = np.zeros(self.nvar, dtype=np.int32)
        for vari in allowed_input_token:
            if vari >= 0 and vari < len(free_input_tokens):
                free_input_tokens[vari] = 1
        Program.task.set_allowed_inputs(free_input_tokens)
        print("set allow input tokens.....")

        if library_disallowed_input_token != -1:
            for vari in library_disallowed_input_token:
                if vari >= 0 and vari < len(free_input_tokens):
                    free_input_tokens[vari] = 0
        self.library.set_allowed_input_tokens(free_input_tokens)
        print("For library:", self.library.allowed_tokens, self.library.allowed_input_tokens)
        print("For data loader:", Program.task.allowed_input, Program.task.fixed_column)

    def print_all_populations(self):
        for vari in self.populations:
            print(f"vars={vari}")
            for pr in self.populations[vari]:
                print("\t", pr.__getstate__())

    def print_final_hofs(self):
        # selected_vars = None
        print(self.hofs.keys())
        for key in self.hofs:
            if len(key.cur) == self.nvar + 1:
                selected_vars = key
                print("\n\n")
                print('%' * 20)
                print(f"vars={selected_vars}")
                print('%' * 20)
                if selected_vars in self.hofs:

                    new_hof = sorted(self.hofs[selected_vars], reverse=True, key=attrgetter('r'))
                    for pr in new_hof:
                        print("\t", pr.__getstate__())
                        pr.task.rand_draw_X_non_fixed()
                        print('\tvalidate r=', pr.task.reward_function(pr))
                        pr.task.print_reward_function_all_metrics(pr)
                        pr.print_expression()


def print_prs(prs):
    for pr in prs:
        print('        ' + str(pr.__getstate__()), end="\t")
        pr.print_expression()
    print("")
