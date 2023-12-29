import numpy as np

class GPHelper(object):
    """
    Function class for genetic programming.
    """

    # static variables
    library = None

    def mate(self, a, b):
        """
            a and b are two program objects. apply cross over of two program trees; find two subtrees
            within allowed_change_tokens then swap them
        """
        a_allowed = a.allow_change_pos()
        b_allowed = b.allow_change_pos()

        if len(a_allowed) == 0 or len(b_allowed) == 0:
            return

        a_start = np.random.choice(a_allowed)
        b_start = np.random.choice(b_allowed)

        a_end = a.subtree_end(a_start)
        b_end = b.subtree_end(b_start)

        na_tokens = np.concatenate((a.tokens[:a_start],
                                    b.tokens[b_start:b_end],
                                    a.tokens[a_end:]))
        nb_tokens = np.concatenate((b.tokens[:b_start],
                                    a.tokens[a_start:a_end],
                                    b.tokens[b_end:]))
        na_vf = a.vf

        na_allow = np.concatenate((a.allow_change_tokens[:a_start],
                                   b.allow_change_tokens[b_start:b_end],  # Question: should it be all ones here?
                                   a.allow_change_tokens[a_end:]))
        nb_allow = np.concatenate((b.allow_change_tokens[:b_start],
                                   a.allow_change_tokens[a_start:a_end],
                                   b.allow_change_tokens[b_end:]))
        nb_vf = b.vf
        a.__init__(na_tokens, na_allow)
        b.__init__(nb_tokens, nb_allow)
        a.remove_r_evaluate()
        b.remove_r_evaluate()
        a.set_self_vf()
        b.set_self_vf()
        a.update_with_other_vf(na_vf)
        b.update_with_other_vf(nb_vf)

        # avf = copy.copy(a.vf)
        # bvf = copy.copy(b.vf)
        # a.update_with_other_vf(bvf)
        # b.update_with_other_vf(avf)

    def muNewVariable(self, p, maxdepth, verbose):
        """
        expand a (random) summary constant with a new variable.
        """

        #  How to know a summary constant?
        if not p.freezed:
            # print("Not freezed {}".format(p))
            return
        # 1. get the list of summary constants from p
        leaf_set = p.summary_constant_pos()
        if len(leaf_set) == 0:
            # print("No summary constants in {} with leaf {}".format(p, leaf_set))
            return
        # 2. pick a new variable for the program.
        t_idx = np.random.choice(np.array(leaf_set))
        if not p.pick_new_random_vf(verbose=verbose):
            # print("No new variable to be inserted {} {}".format(p, p.cur_node))
            return
        self._set_library_allowed_input_tokens(p, p.vf[-1], verbose=True)
        new_tree = np.array(self.gen_full(maxdepth))
        print('muNewVariable', new_tree)
        np_tokens = np.concatenate((p.tokens[:t_idx], new_tree, p.tokens[(t_idx + 1):]))
        np_allow = np.insert(p.allow_change_tokens, t_idx, np.ones(len(new_tree) - 1, dtype=np.int32))

        p.__init__(np_tokens, np_allow)
        # the program is not freezed after calling __init__
        p.remove_r_evaluate()
        p.set_self_vf()

    def gen_full(self, maxdepth):
        """
            generate a full program tree recursively (represented in token indices in library)
        """
        if maxdepth == 1:
            allowed_pos = [t for t in self.library.tokens_of_arity[0] if self.library.allowed_tokens[t] > 0]
            t_idx = np.random.choice(allowed_pos)
            return [t_idx]
        else:
            allowed_pos = self.library.allowed_tokens_pos()
            t_idx = np.random.choice(allowed_pos)

            arity = self.library.tokens[t_idx].arity
            tree = [t_idx]
            for i in range(arity):
                tree.extend(self.gen_full(maxdepth - 1))
            return tree

    def multi_mutate(self, individual, maxdepth, verbose=True):
        """Randomly select one of four types of mutation."""
        self._set_library_allowed_input_tokens(individual, individual.vf, verbose=False)
        v = np.random.randint(0, 5)
        if v == 0:
            self.mutUniform(individual, maxdepth)
        elif v == 1:
            self.mutNodeReplacement(individual)
        elif v == 2:
            self.mutInsert(individual, maxdepth)
        elif v == 3:
            self.mutShrink(individual)
        elif v == 4:
            self.muNewVariable(individual, maxdepth, verbose)
        if individual.vf == None or len(individual.vf) == 0:
            print('No vf', individual)

    def mutUniform(self, p, maxdepth):
        """
            find a leaf node (which allow_change_tokens == 1), replace the node with a gen_full tree of maxdepth.
        """
        old_vf = p.vf
        leaf_set = []
        for i, token in enumerate(p.traversal):
            if p.allow_change_tokens[i] > 0 and token.arity == 0:
                leaf_set.append(i)
        if len(leaf_set) == 0:
            return
        t_idx = np.random.choice(np.array(leaf_set))

        new_tree = np.array(self.gen_full(maxdepth))

        np_tokens = np.concatenate((p.tokens[:t_idx], new_tree, p.tokens[(t_idx + 1):]))
        np_allow = np.insert(p.allow_change_tokens, t_idx, np.ones(len(new_tree) - 1, dtype=np.int32))

        p.__init__(np_tokens, np_allow)
        p.remove_r_evaluate()
        p.set_vf(old_vf)

    def mutNodeReplacement(self, p):
        """
        find a node and replace it with a node of the same arity.
        """
        old_vf = p.vf
        allowed_pos = p.allow_change_pos()
        if len(allowed_pos) == 0:
            return
        a_idx = allowed_pos[np.random.randint(0, len(allowed_pos))]
        arity = p.traversal[a_idx].arity

        allowed_pos = [t for t in self.library.tokens_of_arity[arity] if self.library.allowed_tokens[t] > 0]
        t_idx = np.random.choice(allowed_pos)

        p.tokens[a_idx] = t_idx
        p.__init__(p.tokens, p.allow_change_tokens)
        p.remove_r_evaluate()
        p.set_vf(old_vf)

    def mutInsert(self, p, maxdepth):
        """
            insert a node at a random position, the original subtree at the location
            becomes one of its subtrees.
        """
        old_vf = p.vf
        insert_pos = np.random.randint(0, len(p.tokens))
        subtree_start = insert_pos
        subtree_end = p.subtree_end(subtree_start)

        # generate the new root node

        # more efficient implementation
        non_term_allowed = self.library.allowed_non_terminal_tokens_pos()
        t_idx = np.random.choice(non_term_allowed)

        root_arity = self.library.tokens[t_idx].arity

        which_old_tree = np.random.randint(0, root_arity)

        # generate other subtrees
        np_tokens = np.concatenate((p.tokens[:subtree_start], np.array([t_idx])))
        np_allow = np.concatenate((p.allow_change_tokens[:subtree_start], np.array([1])))

        for i in range(root_arity):
            if i == which_old_tree:
                np_tokens = np.concatenate((np_tokens,
                                            p.tokens[subtree_start:subtree_end]))
                np_allow = np.concatenate((np_allow,
                                           p.allow_change_tokens[subtree_start:subtree_end]))
            else:
                subtree = np.array(self.gen_full(maxdepth - 1))
                np_tokens = np.concatenate((np_tokens, subtree))
                np_allow = np.insert(np_allow, -1, np.ones(len(subtree), dtype=np.int32))

        # add the rest
        np_tokens = np.concatenate((np_tokens, p.tokens[subtree_end:]))
        np_allow = np.concatenate((np_allow, p.allow_change_tokens[subtree_end:]))

        p.__init__(np_tokens, np_allow)
        p.remove_r_evaluate()
        p.set_vf(old_vf)

    def mutShrink(self, p):
        """
            delete a node (which allow_change_tokens == 1), use one of its child to replace
            its position.
        """
        old_vf = p.vf
        allowed_pos = p.allow_change_pos()
        if len(allowed_pos) == 0:
            return
        a_idx = allowed_pos[np.random.randint(0, len(allowed_pos))]
        arity = p.traversal[a_idx].arity

        # print('arity=', arity)
        if arity == 0:
            # replace it with another node arity == 0 (perhaps not this node; but it is OK).
            self.mutNodeReplacement(p)
        else:
            a_end = p.subtree_end(a_idx)

            # get all the subtrees
            subtrees_start = []
            subtrees_end = []
            k = a_idx + 1
            while k < a_end:
                k_end = p.subtree_end(k)
                subtrees_start.append(k)
                subtrees_end.append(k_end)
                k = k_end

            # pick one of the subtrees, and re-assemble
            sp = np.random.randint(0, len(subtrees_start))

            np_tokens = np.concatenate((p.tokens[:a_idx],
                                        p.tokens[subtrees_start[sp]:subtrees_end[sp]],
                                        p.tokens[a_end:]))
            np_allow = np.concatenate((p.allow_change_tokens[:a_idx],
                                       p.allow_change_tokens[subtrees_start[sp]:subtrees_end[sp]],
                                       p.allow_change_tokens[a_end:]))
            p.__init__(np_tokens, np_allow)
            p.remove_r_evaluate()
            p.set_vf(old_vf)

    def _set_library_allowed_input_tokens(self, p, allowed_input_token, verbose=False):
        """Input is a set of free input variables"""

        free_input_tokens = np.zeros(p.n_var, dtype=np.int32)
        if allowed_input_token:
            for vari in allowed_input_token:
                free_input_tokens[vari] = 1
        self.library.set_allowed_input_tokens(free_input_tokens)
        if verbose:
            print("For library:", self.library.allowed_tokens, self.library.allowed_input_tokens)
