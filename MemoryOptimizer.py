from collections import defaultdict
import numpy as np

class MemoryOptimizer():
    def __init__(self, graph=None, opt_level=1, verbose=0):
        # E.g. opt_level=1 means heuristic, opt_level=2 -- integer optimizer.
        # In the whole class we ignore controlled_inputs, since they are irrelevant for the memory.
        assert(opt_level == 1)
        self.opt_level = opt_level
        self.verbose = verbose
        self.store_activations_set = set()
        self.cache = {}

#         self.build_internal_graph(graph)
#         self.find_save_points()
#         self.sort_topologically()


    def build_internal_graph(self, graph=None):
        # TODO: precompute costs for all nodes.
        if graph is None:
            graph = ops.get_default_graph()

        if hasattr(self, 'my_graph'):
            return

        # TODO: exclude starting nodes from the graph and add them to the chosen set.
        self.operations_to_nodes = {}
        for idx, op in enumerate(graph.get_operations()):
            self.operations_to_nodes[op] = idx
            # self.my_graph[idx] should be something with attributes.
            self.my_graph[op] = lambda: 0
            self.my_graph[op].parents = []
            self.my_graph[op].children = []
        self.my_graph = [None] * len(graph.get_operations())
        self.starting_nodes = []
        for op in enumerate(graph.get_operations()):
            node = self.operations_to_nodes(op)
            children = [self.operations_to_nodes(x.op) for x in op.inputs]
            self.my_graph[node].children = children
            for child in children:
                self.my_graph[child].parents.append(node)


    def sort_topologically(self):
        if hasattr(self, 'topological_order'):
            return
        levels_by_name = [None] * len(self.my_graph)
        names_by_level = defaultdict(list)

        def walk_depth_first(node):
            if levels_by_name[node] is not None:
                return levels_by_name[node]
            children = self.my_graph[node].children
            level = 0 if not children else (1 + max(walk_depth_first(child) for child in children))
            levels_by_name[node] = level
            names_by_level[level].append(node)
            return level

        for node in range(len(self.my_graph)):
            walk_depth_first(node)

        self.topological_order = []
        for level in range(len(names_by_level)):
            self.topological_order += names_by_level[level]
        self.topological_order = self.topological_order[::-1]


    def find_save_points(self):
        if self.opt_level == 1:
            self.find_save_points_heuristic()
        else:
            self.find_save_points_integer_prog()


    def find_save_points_integer_prog(self):


    def find_save_points_heuristic(self):
#         TODO: use binary search.
        best = np.inf
        for budget in range(1, 20):
            curr_usage, curr_points = self._find_save_points_budget(budget)
            if curr_usage < best:
                best = curr_usage
                self.store_activations_set = curr_points



    def _find_save_points_budget(self, budget):
        # Copy starting_nodes points to not to corrupt them.
        node_to_group = [None] * len(self.my_graph)
        group_to_nodes = []
        group_cost = []
        chosen = set()
        chosen_cost = 0
        groups_number = 0
        for node in self.topological_order:
            active_parents = [x for x in self.my_graph[node].parents if x not in chosen]
            # TODO: find unique groups!
            active_parent_groups_cost = [group_cost[node_to_group[x]] for x in active_parents]
            curr_group_cost = self.node_cost(node)
            sort_idx = np.argsort(active_parent_groups_cost)
            stopped = len(sort_idx)
            for i, parent_idx in enumerate(sort_idx):
                if curr_group_cost + active_parent_groups_cost[parent_idx] <= budget:
                    curr_group_cost += active_parent_groups_cost[parent_idx]
                else:
                    stopped = i
                    break
            if len(active_parent_groups_cost) == 0:
                # No active parents, start a new group.
                node_to_group[node] = groups_number
                group_to_nodes.append([node])
                group_cost.append(self.node_cost(node))
                groups_number += 1
            else:
                if stopped == 0:
                    # We can't add the node to any group, lets chose current node.
                    assert(self.node_cost(node) + min(active_parent_groups_cost) > budget)
                    chosen.add(node)
                    chosen_cost += self.node_cost(node)
                else:
                    # Merge all the groups corresponding to the parents
                    # active_parents[sort_idx[:stopped]]
                    # TODO: use data structure that allows fast groups merging.
                    merged_group_idx = node_to_group[active_parents[sort_idx[0]]]
                    group_cost[merged_group_idx] = curr_group_cost
                    node_to_group[node] = merged_group_idx
                    group_to_nodes[merged_group_idx].append(node)
                    for parent_idx in sort_idx[1:stopped]:
                        curr_group_nodes = group_to_nodes[node_to_group[active_parents[parent_idx]]]
                        group_to_nodes[merged_group_idx] += curr_group_nodes
                        for x in curr_group_nodes:
                            node_to_group[x] = merged_group_idx
                    for parent_idx in sort_idx[stopped:]:
                        chosen.add(active_parents[parent_idx])
                        chosen_cost += self.node_cost(active_parents[parent_idx])
                        group_cost[node_to_group[active_parents[parent_idx]]] -= self.node_cost(active_parents[parent_idx])


        return chosen_cost + max(group_cost), chosen


    def node_cost(self, node):
#         TODO: use memory estimate here
        return 1


    def recomputed_op(self, op):
#         if op in self.store_activations_set:
#             return op

        if op in self.cache:
            return self.cache[op]

        if self.verbose:
            print('copying ' + op.name)
        inputs = self.recomputed_inputs(op)
        # Copy op.
        op_node_def = copy.deepcopy(op.node_def)
        op_node_def.name = op_node_def.name + '_copy'
        op_def = copy.deepcopy(op._op_def)
        op_copy = Operation(op_node_def, op._graph, inputs, output_types=op._output_types, input_types=op._input_types,
                            control_inputs=op._control_inputs, original_op=op, op_def=op_def)
        #Use Graph's hidden methods to add the op.
        ops.get_default_graph()._add_op(op_copy)
        ops.get_default_graph()._record_op_seen_by_control_dependencies(op_copy)
        for device_function in reversed(ops.get_default_graph()._device_function_stack):
            op_copy._set_device(device_function(op_copy))

        self.cache[op] = op_copy

        return op_copy


    def recomputed_inputs(self, op):
        # For an operation return the list of its inputs.
        # The function copies parts of the computational graph to recompute the result
        # of each op which is not in self.store_activations_set
        inputs = []
        for x in op.inputs:
            if x in self.store_activations_set:
                inputs.append(x)
            else:
                input_op = self.recomputed_op(x.op)
                output_idx = x.op.outputs.index(x)
                inputs.append(input_op.outputs[output_idx])
        return inputs
