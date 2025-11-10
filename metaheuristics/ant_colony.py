"""
Ant Colony Optimization (ACO/ABACO) for Feature Selection
=========================================================

This module implements the Ant-Based Ant Colony Optimization (ABACO) algorithm
specifically adapted for binary feature selection problems.

ACO is inspired by the foraging behavior of ants using pheromone trails to
find optimal paths. In feature selection, ants construct feature subsets by
probabilistically selecting features based on pheromone levels and heuristic information.

Algorithm Overview:
------------------
1. Initialize pheromone levels on feature connections
2. For each iteration:
   a. Each ant constructs a feature subset probabilistically
   b. Evaluate fitness of all ant solutions
   c. Update pheromones based on solution quality
   d. Apply pheromone evaporation
3. Return best feature subset found

Key Components:
--------------
- **Interconnection**: Manages pheromone matrix between features
- **Ant**: Constructs feature subsets using probabilistic rules
- **AntColony**: Coordinates ant population and pheromone updates

Key Features:
------------
- Binary feature selection representation
- Pheromone-based indirect communication
- Heuristic information (mutual information, correlation)
- Min-max pheromone bounds to prevent stagnation
- Elitist pheromone updates

References:
----------
Dorigo, M., & Stützle, T. (2004). Ant colony optimization. MIT press.
Kabir, M. M., Shahjahan, M., & Murase, K. (2012). A new hybrid ant colony
optimization algorithm for feature selection. Expert Systems with Applications, 39(3), 3747-3763.

Author: ASP-ML Team
License: MIT
Version: 1.0.0
"""

import numpy as np
import random
from datetime import datetime
from .base import Metaheuristic


class interconnection:
    """
    Manages pheromone levels between feature pairs in ACO.
    
    Represents the connection between two features with a 2×2 pheromone matrix
    encoding selection/non-selection combinations.
    
    Attributes:
    ----------
    tau_matrix : list[list[float]]
        2×2 matrix storing pheromone levels:
        - tau_matrix[0][0]: Neither feature selected
        - tau_matrix[0][1]: Feature col selected only
        - tau_matrix[1][0]: Feature row selected only  
        - tau_matrix[1][1]: Both features selected
    row : str
        Index of first feature in connection
    col : str
        Index of second feature in connection
    
    Notes:
    -----
    Pheromone values are bounded by tau_min and tau_max to prevent
    premature convergence or complete exploration loss.
    """
    def __init__(self, tau_init, row, col):
        """
        Initialize feature interconnection with pheromone values.
        
        Parameters:
        ----------
        tau_init : float
            Initial pheromone level for all matrix entries
        row : int
            Index of first feature
        col : int
            Index of second feature
        """
        self.tau_matrix = [[tau_init, tau_init], 
                            [tau_init, tau_init]]
        self.row = str(row)
        self.col = str(col)

    def set_tau_values(self, a, b, tau_value):
        self.tau_matrix[a][b] = tau_value

    def set_eta_values(self, a, b, eta_value):
        self.eta_matrix[a][b] = eta_value
    
    def decrease_tau_value(self, decrease_constant, tau_min, tau_max):
        for i in range(len(self.tau_matrix)):
            for j in range(len(self.tau_matrix[i])):
                inf = float((1 - decrease_constant) * self.tau_matrix[i][j])
                if inf < tau_min:
                    self.tau_matrix[i][j] = tau_min
                elif inf > tau_max:
                    self.tau_matrix[i][j] = tau_max
                else:
                    self.tau_matrix[i][j] = inf
                
    def update_type(self, type_update):
        self.type_update = type_update
    
    def get_type_update(self):
        return self.type_eta_update
    
    def get_tau_values(self, a, b):
        return float(self.tau_matrix[a][b])

    def get_tau_matrix(self):
        return self.tau_matrix
    
    def __str__(self):
        return self.row + ' ' + self.col


class Ant(object):
    """
    Individual ant agent that constructs feature subsets probabilistically.
    
    Each ant traverses the feature space, selecting features based on pheromone
    trails and heuristic information, building a candidate feature subset.
    
    Attributes:
    ----------
    id : int
        Unique identifier for this ant
    tabu : list
        List of already visited/selected features
    tour_complete : bool
        Flag indicating if subset construction is complete
    possible_locations : list
        All available features that can be selected
    binary_possible_locations : list
        Binary representation of selected features (1=selected, 0=not)
    pheromone_map : dict
        Reference to global pheromone matrix
    alpha : float
        Pheromone importance weight (typically 1-2)
    beta : float
        Heuristic information importance weight (typically 2-5)
    location : int
        Current feature being considered
    
    Methods:
    -------
    run()
        Execute ant's feature selection process
    choose_path()
        Probabilistically select next feature based on pheromones and heuristic
    """
    def __init__(self, id, start_location, possible_locations, pheromone_map, alpha, beta, _heuristic_info, first_pass=False):
        """
        Initialize ant agent for feature subset construction.
        
        Parameters:
        ----------
        id : int
            Unique ant identifier
        start_location : int
            Initial feature index to begin construction
        possible_locations : list
            All available feature indices
        pheromone_map : dict
            Global pheromone matrix shared across colony
        alpha : float
            Pheromone trail importance (α ∈ [0, 5], typically 1)
        beta : float
            Heuristic information importance (β ∈ [0, 10], typically 2-5)
        _heuristic_info : np.ndarray
            Heuristic values for features (e.g., mutual information, correlation)
        first_pass : bool, optional
            If True, uses uniform probabilities for initial pheromone setup
        
        Notes:
        -----
        - Higher α favors exploitation of known good solutions
        - Higher β favors features with better heuristic values
        - Balance between α and β controls exploration vs exploitation
        """
        self.tabu = []
        self.id = id
        self.tour_complete = False
        self.possible_locations = list(possible_locations)
        self.binary_possible_locations = [0] * (len(self.possible_locations))
        self.start_location = start_location
        self.nodes_indexes = list(range(len(self.possible_locations)))
        self.pheromone_map = pheromone_map
        self.first_pass = first_pass
        self.alpha = alpha
        self.beta = beta
        self.location = start_location
        self.heuristic_info = _heuristic_info
        self.coordinates = []
        self.update_route(self.start_location)
        self.binary_possible_locations[start_location] = 1
        
    def find_path(self):
        while self.nodes_indexes:
            next, is_selected = self.pick_path()
            if next == None and is_selected == None:
                break
            self.traverse(next, is_selected)   
        self.tour_complete = True
    
    def __str__(self):
        return str(self.id)

    def update_route(self, new):
        self.tabu.append(new)
        self.nodes_indexes.remove(new)

    def get_coordinates(self):
        return self.coordinates

    def pick_path(self):
        # Calculating tau * eta
        sum_total_l0 = 0.0
        sum_total_l1 = 0.0
        # Current possible edges that nodes can choice. For example, (0,1) to j or (0,0) j
        current_possible_nodes = []
        # A list of probabilities weights
        probability_list = []
        if self.binary_possible_locations[self.location] == 1:
            for index_next_location in self.nodes_indexes:
                interconnection_obj = self.pheromone_map[self.location][index_next_location]

                tau_value = interconnection_obj.get_tau_values(1, 0)
                eta_value = self.heuristic_info['eta_0']
                current_possible_nodes.append((index_next_location, 0))
                no_select = pow(tau_value, self.alpha) * pow(eta_value, self.beta)

                sum_total_l0 += no_select

                # For the sum of l1 
                tau_value = interconnection_obj.get_tau_values(1, 1)
                eta_value = self.heuristic_info['eta_1'][self.possible_locations[index_next_location]]
                current_possible_nodes.append((index_next_location, 1))
                select = pow(tau_value, self.alpha) * pow(eta_value, self.beta)

                sum_total_l1 += select

                probability_list.append(no_select)
                probability_list.append(select)
        else:
            for index_next_location in self.nodes_indexes:
                interconnection_obj = self.pheromone_map[self.location][index_next_location]

                tau_value = interconnection_obj.get_tau_values(0, 1)
                eta_value = self.heuristic_info['eta_1'][self.possible_locations[index_next_location]]
                current_possible_nodes.append((index_next_location, 1))
                select = pow(tau_value, self.alpha) * pow(eta_value, self.beta)

                sum_total_l1 += select

                # For the sum of l1 
                tau_value = interconnection_obj.get_tau_values(0, 0)
                eta_value = self.heuristic_info['eta_0']

                current_possible_nodes.append((index_next_location, 0))
                no_select = pow(tau_value, self.alpha) * pow(eta_value, self.beta)

                probability_list.append(no_select)
                probability_list.append(select)
                sum_total_l0 += no_select

        sum_total = sum_total_l1 + sum_total_l0

        if sum_total != 0.0:
            for i in range(len(probability_list)):
                probability_list[i] = float(probability_list[i] / sum_total)
        else:
            return None, None
        next_node, type_connection = random.choices(current_possible_nodes, probability_list, k=1)[0]
        return next_node, type_connection

    def traverse(self, end, is_selected):
        self.update_route(end)
        self.binary_possible_locations[end] = is_selected
        # print('Binary possibles: ', self.binary_possible_locations)
        self.coordinates.append((self.location, end))
        self.location = end

    def get_route(self):
        if self.tour_complete:
            return [self.possible_locations[i] for i in range(len(self.binary_possible_locations)) if self.binary_possible_locations[i] == 1]
        return None
    
    def get_binary_route(self):
        return self.binary_possible_locations


class AntColony(Metaheuristic):
    """Binary Ant Colony Algorithm"""
    def __init__(self, data, labels, generations, q, ro, alpha, beta, tau_min, tau_max, tau_init, objectiveFunction):
        """
        alpha: Represents the weight of pheromone
        beta: Represents the weight of distance
        generations: Number of iterations
        ant_count: Number of ants running per iteration
        n_best: Number of best ants who deposit pheromones 
        Q: a constant related to the quantity of trail laid by ants
        """
        Metaheuristic.__init__(self)
        self.name = 'ABACO'
        self.setDataSet(data, labels)
        self.ant_count = data.shape[1]
        self.epochs = generations
        self.alpha = alpha
        self.beta = beta
        self.Q = q
        self.ro = ro
        self.epsilon = 1
        self.tau_max = tau_max
        self.tau_init = tau_init
        self.tau_min = tau_min
        self.loss_value = None
        self.time = str(datetime.now().strftime("%d_%m_%Y_%H_%M_%S"))
        self.nodes = [i for i in range(data.shape[1])]
        self.heuristic_info = {'eta_0': None, 'eta_1': {}}
        self.max_auc = None
        self.max_auc_path = None
        self.loss_value = None
        self.recall_value = None
        self.accuracy_value = None
        self.best_generation = None
        self.loss_value = None
        self.first_pass = True
        self.best_generation = 0
        self.objectiveFunction = objectiveFunction
        self.results_history = []
        self.confusion_history = []
    
    def initialize(self):
        self.ant_count = self.train_data.shape[1]
        self.nodes = [i for i in range(self.train_data.shape[1])]
        self.heuristic_info = {'eta_0': None, 'eta_1': {}}
        self.results_history = []
        self.confusion_history = []
        self.max_auc_path = None
        self.max_auc = None
        self.best_generation = 0
        self.first_pass = True
        self.historic = []
    
    def setDataSet(self, data, labels):
        self.train_data = data
        self.train_labels = labels
        self.n_classes = np.unique(labels)
        print(f'Set train dataset: {self.name} - {self.train_data.shape}')

    def calculate_scores(self):
        original_len = len(list(self.nodes))
        for value in self.nodes:
            feature_i = self.train_data[:, value]
            discriminant_numerator = 0
            discriminant_denominator = 0
            results = []
            for k in self.n_classes:
                feature_i_k = [i for i, j in zip(feature_i, self.train_labels) if j == k]
                mean_feature_i = np.mean(feature_i)
                mean_feature_i_k = np.mean(feature_i_k)
                discriminant_numerator += pow((mean_feature_i_k - mean_feature_i), 2)
                # For denominator sum k = 1, c[(1/(Ni^k - 1) * sum(j=1, Ni^k) (xij^k - mean(xi^k))^2)]
                for den_k in self.n_classes:
                    feature_i_k = [i for i, j in zip(self.train_data[:, value], self.train_labels) if j == den_k]
                    mean_feature_k = np.mean(feature_i_k)
                    len_feature_by_class = len(feature_i_k)
                    tempo = 0
                    # pow(xij^k - mean(xi^k)) 
                    for j in feature_i_k:
                        tempo += pow((j - mean_feature_k), 2)
                    tempo = tempo * (1/(len_feature_by_class-1))
                    discriminant_denominator += tempo

                if discriminant_denominator != 0.0 and discriminant_numerator != 0.0:
                    f_score_i = float(discriminant_numerator / discriminant_denominator)
                    results.append(f_score_i)
                else:
                    results.append(0.0)
            f_score = sum(results)
            if f_score != 0.0:
                self.heuristic_info['eta_1'][value] = sum(results)
        self.heuristic_info['eta_0'] = (self.epsilon / len(self.nodes) * sum(self.heuristic_info['eta_1'].values()))
        f_len = len(list(self.heuristic_info['eta_1'].keys()))
        print('>> F Score calculated, features with zero value have been discarded! \n>> Features reduced from ' + str(original_len) + ' to ' + str(f_len))
        self.final_len = len(list(self.heuristic_info['eta_1'].keys()))
    
    def init_matrix(self, n):
        ret = []
        for row in range(n):                                                
            ret.append([interconnection(self.tau_init, row, x) for x in range(n)])
        return ret
    
    def init_ants(self):
        if self.first_pass:
            ant_array = []
            for i in range(self.ant_count):
                start = random.randint(0, self.final_len - 1)
                ant_array.append(Ant(i, start, self.heuristic_info['eta_1'].keys(), self.pheromone_map, self.alpha, self.beta, self.heuristic_info, first_pass=True))
            return ant_array
        else:
            for i in range(self.ant_count):
                start = random.randint(0, self.final_len - 1)
                self.ants[i].__init__(i, start, self.heuristic_info['eta_1'].keys(), self.pheromone_map, self.alpha, self.beta, self.heuristic_info)

    def runProcess(self, q):
        bestSubset = self.run()
        q.put((bestSubset, self.historic, self.name, self.epochs, self.bestScoreValue))
    
    def run(self):
        self.calculate_scores()
        self.set_matrixes()
        t = 0
        while t < self.epochs:
            # This is used to wait until the ants are finished, before update
            for ant in self.ants:
                ant.find_path()
            local_auc_list = []
            for ant in self.ants:
                value = self.objectiveFunction(ant.get_route(), self.train_data, self.train_labels)
                local_auc_list.append(value)
            
            max_auc = max(local_auc_list)
            max_index_auc = local_auc_list.index(max_auc)
            self.historic.append(max_auc)
            accuracy_value = 0
            recall_value = 0
            loss_value = 0
            best_ant = self.ants[max_index_auc]

            if not self.max_auc:
                self.max_auc = max_auc
                self.loss_value = loss_value
                self.recall_value = recall_value
                self.accuracy_value = accuracy_value
                self.best_generation = t
                self.loss_value = loss_value

            if not self.max_auc_path:
                self.max_auc_path = ant.get_route()
                print('>> Init values auc: ', self.max_auc, 'Len subset: ', len(self.max_auc_path), ' in gen ', t)
                
            if max_auc > self.max_auc:
                self.max_auc = max_auc
                self.max_auc_path = ant.get_route()
                self.loss_value = loss_value
                self.recall_value = recall_value
                self.accuracy_value = accuracy_value
                self.best_generation = t
                print('>> New max value: ', max_auc, 'Len subset: ', len(self.max_auc_path), ' in gen ', t)

            if max_auc == self.max_auc:
                if len(ant.get_route()) < len(self.max_auc_path):
                    self.max_auc = max_auc
                    self.max_auc_path = ant.get_route()
                    self.loss_value = loss_value
                    self.recall_value = recall_value
                    self.accuracy_value = accuracy_value
                    self.best_generation = t
                    print('>> Subset updated to min length subset due to the same precision was computed!')
                    print('>> Updated max pre: ', max_auc, ' ', recall_value, ' - Loss: ', loss_value, ' Acc: ', accuracy_value, ' in generation ', t, 'Len subset: ', len(self.max_auc_path))

            self.update_pheromone_map(best_ant, loss_value)
            if self.first_pass: 
                self.first_pass = False
            self.init_ants()
            t += 1
            print('ABACO: Generation: %d - %.5f - %d' % (t, self.max_auc, len(self.max_auc_path)))
        ret = []
        print('Nodes: ', self.nodes)
        print('Max path: ', self.max_auc_path)
        for id in self.max_auc_path:
            ret.append(self.nodes[id])
        print('Finish time: ', str(datetime.now().strftime("%d-%m-%Y %H-%M-%S")))
        self.bestScoreValue = self.max_auc
        self.bestSubset = ret
        return ret

    def set_matrixes(self):
        self.pheromone_map = self.init_matrix(self.final_len)
        print('Shape pheromone: ', len(self.pheromone_map), 'x', len(self.pheromone_map[0]))
        self.ants = self.init_ants()
        print('Ants: ', self.ant_count)
        print('Matrixes started')

    def update_pheromone_map(self, ant, loss_value):
        """
        1) Update the value of self.pheromone_map by caying values give by tau_i_j(t + n) = 1 - ro * t_i_j(t) + delta tau_i_j

        2) Add new pheromones_values from ant_updated_pheromone_map
        """
        binary_ant_route = ant.get_binary_route()
        route_coordinates = ant.get_coordinates()
        # print_pheromone_map(self.pheromone_map)
        for i in range(len(self.pheromone_map)):
            for j in range(len(self.pheromone_map)):
                if i != j:
                    interconnection_obj = self.pheromone_map[i][j]
                    interconnection_obj.decrease_tau_value(self.ro, self.tau_min, self.tau_max)

        for coordinate in route_coordinates:
            x, y = coordinate

            interconnection_obj = self.pheromone_map[x][y]
            x_is_selected = int(binary_ant_route[x])
            y_is_selected = int(binary_ant_route[y])

            tau_value = interconnection_obj.get_tau_values(x_is_selected, y_is_selected)
            tau_value_updated = float(tau_value + (self.Q / 0.10))
            if tau_value_updated > self.tau_max:
                tau_value_updated = self.tau_max
            elif tau_value_updated < self.tau_min:
                tau_value_updated = self.tau_min
            interconnection_obj.set_tau_values(x_is_selected, y_is_selected, tau_value_updated)
        print('>> Pheromone map updated')
