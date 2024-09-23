"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 ___        __                                                _   _ _   _             _             
|_ _|_ __  / _| ___  _ __ ___   __ _ _ __    _ __   __ _ _ __| |_(_) |_(_) ___  _ __ (_)_ __   __ _ 
 | || '_ \| |_ / _ \| '_ ` _ \ / _` | '_ \  | '_ \ / _` | '__| __| | __| |/ _ \| '_ \| | '_ \ / _` |
 | || | | |  _| (_) | | | | | | (_| | |_) | | |_) | (_| | |  | |_| | |_| | (_) | | | | | | | | (_| |
|___|_|_|_|_|  \___/|_| |_| |_|\__,_| .__/  | .__/ \__,_|_|   \__|_|\__|_|\___/|_| |_|_|_| |_|\__, |
  __ _| | __ _  ___  _ __(_) |_| |__|_| __ _|_|                                               |___/ 
 / _` | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \                                                     
| (_| | | (_| | (_) | |  | | |_| | | | | | | | |                                                    
 \__,_|_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_|                                                    
         |___/                                                                                      

METHODS OF THE CLASS:

    * community_partitioning
    * visualize_partitions
    * graph_info
    * map_equation
    * print_modules

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
import pandas as pd
import igraph as ig
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import random

from tqdm import tqdm
from time import sleep

sns.set_theme()


class Infomap:
    def __init__(self):
            pass

    def community_partitioning(self, adjacency_matrix, 
                                pagerank_on=True, 
                                d=0.85, 
                                tol=1e-10, 
                                iterations=2, 
                                shuffle=False,          
                                normalize_weights=True,
                                normalize_weights_after_merging=False,
                                move_out=False,
                                verbose=False): 
        
        """
        method that implements the infomap partition alogrithm.

            inputs: - modules: dictioray,
                    - possible_links: dictionary,
                    - adjacent_matrix: matrix,
                    - pagerank_on: bool,
                    - d: float,
                    - tol: float,
                    - normalize_weights: bool,
                    - normalize_weihts_after_merging: bool,
                    - verbose: bool

        outputs:    - best_modules: dictionary,
                    - final_links: dictonary,
                    - final_partition: dictionary,
        """

        self.adjacency_matrix = adjacency_matrix 
        self.pagerank_on = pagerank_on 
        self.d = d 
        self.tol = tol
        self.iterations = iterations 
        self.shuffle = shuffle
        self.normalize_weights = normalize_weights
        self.normalize_weights_after_merging = normalize_weights_after_merging
        self.move_out = move_out
        self.verbose = verbose


        best_modules = None
        final_links = None
        final_partition = None

        # we remove self interaction from the matrix because they do not contribute for the partitioning
        adjacency_matrix = np.array(adjacency_matrix)
        np.fill_diagonal(adjacency_matrix,0)

        if verbose:
            print("=======================================================================================================================")
            graph_df=graph_info(adjacency_matrix)           
            print(graph_df,"\n")
            print("# Nodes connections:")
            print(initialize_possible_links(graph_df,move_out=move_out))
            print("=======================================================================================================================")
            
            for i in range(iterations):
                print(f"** Iteration: {i} **\n")

                current_modules = copy.deepcopy(best_modules)
                current_links = copy.deepcopy(final_links)
                current_partition = copy.deepcopy(final_partition)

                current_modules, current_links, current_partition = iterative_infomap_partitioning(modules=current_modules,
                                                                                                    possible_links=current_links, 
                                                                                                    adjacent_matrix=adjacency_matrix,
                                                                                                    pagerank_on=pagerank_on,
                                                                                                    d=d,
                                                                                                    tol=tol,
                                                                                                    shuffle=shuffle,
                                                                                                    normalize_weights=normalize_weights,
                                                                                                    normalize_weights_after_merging=normalize_weights_after_merging,
                                                                                                    move_out=move_out,
                                                                                                    verbose=verbose)

                best_modules = current_modules
                final_links = current_links 
                final_partition = current_partition 
                print("=======================================================================================================================")

            print(f"Done! Number of partitions: {len(final_partition.keys())}")

            return best_modules, final_links, final_partition

        else:
            for i in tqdm(range(iterations), desc="Partitioning the network", ascii=False, ncols=75):
                if verbose:
                    print(f"** Iteration: {i} **\n")

                current_modules = copy.deepcopy(best_modules)
                current_links = copy.deepcopy(final_links)
                current_partition = copy.deepcopy(final_partition)


                current_modules, current_links, current_partition = iterative_infomap_partitioning(modules=current_modules,
                                                                                                    possible_links=current_links, 
                                                                                                    adjacent_matrix=adjacency_matrix,
                                                                                                    pagerank_on=pagerank_on,
                                                                                                    d=d,
                                                                                                    tol=tol,
                                                                                                    shuffle=shuffle,
                                                                                                    normalize_weights=normalize_weights,
                                                                                                    normalize_weights_after_merging=normalize_weights_after_merging,
                                                                                                    move_out=move_out,                                                                                                    
                                                                                                    verbose=verbose)
                best_modules = current_modules
                final_links = current_links 
                final_partition = current_partition 

                sleep(0.02)

            print(f"Done! Number of partitions: {len(final_partition.keys())}")

            return best_modules, final_links, final_partition


    def visualize_partitions(self, graph, final_partitions, 
                             layout = "auto", 
                             edge_width = 0.5, 
                             vertex_label = None, 
                             vertex_label_size=8.0,
                             vertex_size = 25, 
                             plot_size = (7,5), 
                             colour_map = "tab10", 
                             title = ""):
    
        """
        method that plots the graph using different colors for nodes that are in different partitions.
        
        inputs: - graph: igraph graph
                - partitions: dictionary, dictionary of partitioed nodes
                - layout: string,
                - edge_width: numeric,
                - vertex_label: list,
                - vertex_size: numeric
                - plot_size: tuple,
                - color_map: string, colours that will be used for the visualization
                - title: string, title of the plot

        output:   - plot of the graph with partitions
        """

        self.graph = graph
        self.final_partitions = final_partitions
        self.layout = layout
        self.edge_width = edge_width
        self.vertex_label = vertex_label
        self.vertex_label_size = vertex_label_size,
        self.vertex_size = vertex_size
        self.plot_size = plot_size
        self.colour_map = colour_map
        self.title = title

        plot_graph_with_partitions(graph, final_partitions, layout=layout, edge_width=edge_width, vertex_label=vertex_label, vertex_label_size=vertex_label_size,vertex_size=vertex_size, plot_size=plot_size, colour_map=colour_map, title=title)


    def graph_info(self, adjacent_matrix, pagerank_on=True, d=0.85, tol=1e-10, normalize_weights=True):
        """
        function that given the ajacency matrix of a graph produces a dataframe containing for each node: 
            * the visiting frequencies for the stationary state for the associate markovian process
            * a list of linked nodes
            * a list of weights for each link 

        input:  - adjacencent_matrix: matrix, adjacent matrix of the graph
                - pagerank: bool, 
                - d: float, damping factor
                - tol: float, tolerance for the pagerank algorithm
                - normalize_weights: bool, normalization of the weights for each node connection
                
        output:   - graph_info_df: dataframe
        """

        self.adjacent_matrix = adjacent_matrix
        self.pagerank_on = pagerank_on
        self.d = d
        self.tol = tol
        self.normaize_weights = normalize_weights

        df = graph_info(adjacent_matrix=adjacent_matrix, pagerank_on=pagerank_on, d=d, tol=tol, normalize_weights=normalize_weights)
        
        return df
    

    def map_equation(self, modules, verbose=False):
        """
        method that evaluets the map equation given a dictionary of partitions.

        inputs:   - modules: dictionary, dictionary of modules to evaluate
                - verbose: bool, control parameter for debugging

        output:   - bits: float, result of the map equation
        """

        self.modules = modules
        self.verbose = verbose

        bits = evaluete_map_equation(modules, verbose=verbose)

        return bits


    def print_modules(self, modules):
        """
        auxilary method for clear modules visualization

        intput: - modules: dictionary, dictionary of module
        """

        self.modules = modules

        print_modules(modules)
    
#===================================================================================================================================
"""
AUXILIARY FUNCTIONS OF THE CLASS:

    1. COLLECTING INFORMATIONS
        - relative_weights
        - normalize matrix
        - pagerank
        - graph_info

    2. INITIALIZE MODULES
        - initialize_modules
        - print_modules

    3. MAP EQUATION
        - evaluate_map_equation

    4. BINARY MERGING ALGORITHM
        - initialize_possible_links
        - merging_possible_links
        - merge_modules

    5. ITERATIVE INFOMAP ALPGORITHM
        - iterative_infomap_partitioning
    
    6. VISUALIZE PARTITIONS
        - plot_graph_with_partitions
"""


# 1. COLLECTING INFORMATIONS:
#-----------------------------------------------------------------------------------------------------------------------------------

def relative_weights(M):
    """
    function that evaluetes the stationary state for a markovian process given the transition matrix
    input:  - M: matrix, transition matrix

    output: - v: list, list of visiting frequecy of each node

    """        

    total_weight = np.sum([np.sum(M[i]) for i in range(M.shape[1])])
    v = [np.sum(M[i])/total_weight for i in range(M.shape[1])]

    return v


def normalize_matrix(M, axis=0):
    """
    function that normalizes a matrix
    input:  - M: matrix
            - axis:int, * 0: normalization by cols
                        * 1: normalizaion by rows

    output: - nomalized matrix
    """
    
    sums = np.sum(M, axis=axis)
    # avoid division by 0
    sums[sums == 0] = 1

    return M / sums


def pagerank(M, d=0.85, tol=1e-10, axis=0):
    """ 
    function that evaluetes the stationary state for a markovian process with teleportation given the transition matrix

    input:  - M: matrix, trasnition matrix
            - d: float, dumping factor
            - tol: float, tolerance parameter
            - axis: int,   * 0: normalization by cols
                            * 1: normalizaion by rows

    output: - v: list, list of visiting frequencies of each node
    """

    M = normalize_matrix(M, axis=axis)  
    N = M.shape[1]  
    w = np.ones(N) / N  
    M_hat = d * M  
    
    v = M_hat @ w + (1 - d) / N  
    
    while np.linalg.norm(w - v) >= tol:
        w = v
        v = M_hat @ w + (1 - d) / N
    
    return v


def graph_info(adjacent_matrix, pagerank_on=True, d=0.85, tol=1e-10, normalize_weights=True):
    """
    function that given the ajacency matrix of a graph produces a dataframe containing for each node: 
          * the visiting frequencies for the stationary state for the associate markovian process
          * a list of linked nodes
          * a list of weights for each link 

    input:    - adjacencent_matrix: matrix, adjacent matrix of the graph
              - pagerank: bool, 
              - d: float, damping factor
              - tol: float, tolerance for the pagerank algorithm
              - normalize_weights: bool, normalization of the weights for each node connection
              
    output:   - graph_info_df: dataframe
    """

    nodes_list =  np.arange(adjacent_matrix.shape[0])
    links_for_node = [list(i) for i in np.column_stack(adjacent_matrix)]  
    weigths = []

    for node in nodes_list:
        links = adjacent_matrix[:,node]
        weigths.append([links[i] for i in range(len(links)) if links[i]!=0]) 
        
    if pagerank_on is True:
        stationary_solution = pagerank(adjacent_matrix,d=d,tol=tol)

    else:
        stationary_solution = relative_weights(adjacent_matrix)

    nodes_links_dict = {}
    for node, links in zip(nodes_list,links_for_node):
        nodes_links_dict[node] = links

    linked_nodes = []
    for i in range(len(nodes_links_dict)):
        possible_links = nodes_links_dict[i]
        indeces = [i for i in range(len(possible_links)) if possible_links[i] != 0]
        linked_nodes.append(indeces)

    if normalize_weights:
        weigths = [[i / sum(node_weights) if sum(node_weights) != 0 else 0 for i in node_weights] for node_weights in weigths]

    graph_info_df = pd.DataFrame({"node": nodes_list, "freq": stationary_solution, "linked_nodes":linked_nodes, "weights":weigths})

    return graph_info_df


# 2. INITIALIZE MODULES
#-----------------------------------------------------------------------------------------------------------------------------------

def initialize_modules(graph_df):
    """
    function that generate a dictionary of matrix corresponding to each network's module
    the initial prartition consider each node in the network as a single module
    
        * 'nodes modules':
            they are named from 0 to n-1 for each of the n nodes in the network
            each module is a matrix where the first row is the list of nodes presented in the module
            the first element of the fist row is '-1' and represent the 'exit node'
            the second row is the list of the frequencies for the stady state of the associated markovian process 

    input:    - grap_df: dataframe, dataframe produced by graph_info()

    output:   - initial_modules: dictionary, dictionary of the starting modules
    """
    node_freq = list(graph_df["freq"])

    initial_modules = {}

    for i,j in enumerate(node_freq):
        initial_modules[i] = np.array([[-1,i],[j,j]])

    return initial_modules


def print_modules(modules):
    """
    auxilary function to visualize modules
    
    input:    - moduels: dictonary
    """ 
    for module_id, module in modules.items():
        print("MODULE: {}".format(module_id))
        print(module,"\n")


# 3. MAP EQUATION
#-----------------------------------------------------------------------------------------------------------------------------------

def evaluete_map_equation(modules, verbose=False):
    """
    function that evaluets the map equation

    inputs:   - modules: dictionary, dictionary of modules to evaluate
              - verbose: bool, control parameter for debugging

    output:   - bits: float, result of the map equation
    """

    exit_rates = [modules[i][1,0] for i in modules.keys()]
    exit_rates = [0 if i is None else i for i in exit_rates]
    exit_sum = np.sum(exit_rates)

    nodes_rates = [list(modules[i][1,1:]) for i in modules.keys()]
    flatten_nodes_rates = [i for sublist in nodes_rates for i in sublist ]
    exit_and_nodes_rate = [np.sum(modules[i][1,:]) for i in modules.keys()]

    if exit_sum == 0:
        term_1 = 0
        term_2 = 0

    else:
        term_1 = exit_sum * np.log2(exit_sum)

        log_exit_rates = [0 if i == 0 else np.log2(i) for i in exit_rates]
        term_2 = - 2 * np.sum(np.multiply(exit_rates,log_exit_rates))
    
    log_flatten_nodes_rates = [0 if i == 0 else np.log2(i) for i in flatten_nodes_rates]
    term_3 = - np.sum(list(np.multiply(flatten_nodes_rates,log_flatten_nodes_rates)))

    log_exit_and_nodes_rates = [0 if i == 0 else np.log2(i) for i in exit_and_nodes_rate]
    term_4 = + np.sum(list(np.multiply(exit_and_nodes_rate,log_exit_and_nodes_rates)))

    bits = term_1 + term_2 + term_3 + term_4

    if verbose:
        print("term 1: {}".format(term_1))
        print("term 2: {}".format(term_2))
        print("term 3: {}".format(term_3))
        print("term 4: {}\n".format(term_4))
        print("bits: {}".format(np.round(bits,2)))
        print("------------------------------------------------")

    return bits


# 4. BINARY MERGING ALGORITHM
#-----------------------------------------------------------------------------------------------------------------------------------

def initialize_possible_links(graph_df, move_out=False):
    """
    function that initialize a dictionary of possible links between nodes.
    the structure is: {source_node_1: [(end_node_1, edge_weight_1), (end_node_2, edge_weight_2), ...]...}
    
    input:  - graph df: dictionary, dictionary produced by graph_info()
            - move_out: bool, condition for buil the dictionary using the links that go out of the module

    output: - possible_links: dictionary, dictionary of connections between modules
    """

    if move_out:
        possible_links = {}
        
        for _, row in graph_df.iterrows():
            end_node = row["node"]
            for source_node, weight in zip(row["linked_nodes"], row["weights"]):
                if source_node not in possible_links:
                    possible_links[source_node] = []
                possible_links[source_node].append((end_node, weight))

            possible_links = {k: possible_links[k] for k in sorted(possible_links)}

        # add nodes with no connections/not pointed by other nodes
        for i in graph_df["node"]:
            if i not in possible_links.keys():
                possible_links[i] = []
    
            possible_links = dict(sorted(possible_links.items()))

    # in this case we initialize the links that go in the target node
    else:
        possible_links = {i: list(zip(graph_df["linked_nodes"][i], graph_df["weights"][i])) for i in range(len(graph_df["linked_nodes"]))}


    return possible_links


def merging_possible_links(possible_links, target_id, merging_id, normalization=False):
    """
    Function that produces a dictionary of possible links between modules after the merging.
    
    Inputs: 
        - possible_links: dictionary, dictionary of possible links from the module
        - target_id: int, the target module ID
        - merging_id: int, the module ID to be merged
        - normalization: boolean, flag to indicate if weights should be normalized

    Output: 
        - new_possible_links: dictionary, updated possible links after merging
    """
    
    # check if merging_id is reachable from target_id
    if merging_id not in [node for node, _ in possible_links[target_id]]:
        raise ValueError("Merging module unreachable from target module")
    
    new_possible_links = copy.deepcopy(possible_links)
    
    # get the links for target_id and merging_id
    target_links = new_possible_links[target_id]
    merging_links = new_possible_links[merging_id]

    # create a dictionary to sum weights for combined links
    combined_links = {}
    for node, weight in target_links + merging_links:
        if node != target_id and node != merging_id:
            if node in combined_links:
                combined_links[node] += weight
            else:
                combined_links[node] = weight

    if normalization:
        # normalize the weights for target_id
        total_weight = sum(combined_links.values())
        if total_weight > 0:
            combined_links = {node: weight / total_weight for node, weight in combined_links.items()}

    # update the links for target_id with combined  weights
    new_possible_links[target_id] = [(node, weight) for node, weight in combined_links.items()]

    # update the reciprocal links and normalize if needed
    for node, weight in combined_links.items():
        new_links = []
        merged_weight = 0
        for linked_node, linked_weight in new_possible_links[node]:
            if linked_node == merging_id or linked_node == target_id:
                merged_weight += linked_weight
            else:
                new_links.append((linked_node, linked_weight))

        if normalization:
            if merged_weight > 0:
                total_linked_weight = sum(w for n, w in new_possible_links[node] if n != target_id and n != merging_id)
                total_linked_weight += merged_weight
                if total_linked_weight > 0:
                    new_links.append((target_id, merged_weight / total_linked_weight))
            else:
                new_links.append((target_id, weight))
        else:
            if merged_weight > 0:
                new_links.append((target_id, merged_weight))
            else:
                new_links.append((target_id, weight))

        # normalize the weights for reciprocal links if needed
        if normalization:
            total_weight_node = sum(w for _, w in new_links)
            if total_weight_node > 0:
                new_links = [(n, w / total_weight_node) for n, w in new_links]

        new_possible_links[node] = new_links

    # remove self connections for target_id
    new_possible_links[target_id] = [item for item in new_possible_links[target_id] if item[0] != target_id]

    # replace references to merging_id with target_id in all nodes
    for node, links in new_possible_links.items():
        updated_links = []
        for linked_node, weight in links:
            if linked_node == merging_id:
                linked_node = target_id
            updated_links.append((linked_node, weight))
        new_possible_links[node] = updated_links

    # remove the merged module id from the dictionary
    new_possible_links.pop(merging_id)

    # sort the links for each node by the link id
    for node in new_possible_links:
        new_possible_links[node] = sorted(new_possible_links[node], key=lambda x: x[0])

    return new_possible_links


def merge_modules(modules, possible_links, target_module, graph_df, 
                  pagerank_on=True, 
                  d=0.85, 
                  normalize_weights_after_merging=False,
                  move_out=False,
                  verbose=False):
    """
    function that merge linked modules according to the minimization of the map equation

        inputs: - modules: dictioray,
                - possible_links: dictionary,
                - target_module: int,
                - graph_df: dataframe,
                - pagerank_on: bool,
                - d: float,
                - normalize_weihts_after_merging: bool,
                - verbose: bool

    outputs:    - best_modules: dictionary,
                - new_possible_links: dictonary,
    """
    
    if not possible_links:
        possible_links = initialize_possible_links(graph_df,move_out=move_out)

    best_modules = copy.deepcopy(modules)
    best_evaluation = evaluete_map_equation(best_modules)
    best_linked_module = None
    new_possible_links = None

    N_nodes = len(graph_df["node"])
    total_links = np.sum([len(links) for links in possible_links.values()])
    total_weights = np.sum([np.sum(i) for i in graph_df["weights"]])


    if verbose:
        print(f"target module: {target_module}")
        print(f"possible links: {possible_links[target_module]}\n")

    if possible_links[target_module]:
        for merge_id, _ in possible_links[target_module]:
            current_possible_links = merging_possible_links(possible_links, target_module, merge_id, normalize_weights_after_merging)
            current_modules = copy.deepcopy(modules)

            # merging freqs
            new_node_list = list(modules[target_module][0]) + list(modules[merge_id][0][1:])
            new_frq_list = list(modules[target_module][1]) + list(modules[merge_id][1][1:])

            current_modules[target_module] = np.array([new_node_list, new_frq_list])

            # ordering the matrix according to the nodes name
            sorted_indices = np.argsort(current_modules[target_module][0])
            current_modules[target_module] = current_modules[target_module][:, sorted_indices]

            # update the exit rate
            modules_weights = [weight for __, weight in current_possible_links[target_module]]  

            if pagerank_on:
                N_module = len(current_modules[target_module][0]) - 1

                modules_freqs = new_frq_list[1:]
                weighted_freqs = [freq*weight for freq,weight in zip(modules_freqs,modules_weights)]

                current_modules[target_module][1, 0] = (1 - d) * (N_nodes - N_module) / (N_nodes-1) * np.sum(modules_freqs) + \
                                                        d * np.sum(weighted_freqs)
            else:
                exit_freqs = [1/total_links]*len(current_possible_links[target_module])
                weighted_freqs = [freq*weight/total_weights for freq,weight in zip(exit_freqs,modules_weights)] #####
                current_modules[target_module][1, 0] = np.sum(weighted_freqs)
                

            del current_modules[merge_id]

            evaluation = evaluete_map_equation(current_modules)

            if verbose:
                print(f"merge id: {merge_id}, bits: {evaluation}")
        
            if evaluation < best_evaluation:
                best_modules = current_modules
                best_evaluation = evaluation 
                best_linked_module = merge_id
                new_possible_links = current_possible_links 

        if best_linked_module is not None:
            if best_linked_module not in [node for node, _ in current_possible_links[target_module]]:
                current_possible_links[target_module].append((best_linked_module, 0))
                current_possible_links[target_module] = sorted(current_possible_links[target_module], key=lambda x: x[0])

            for key, value_list in current_possible_links.items():
                if best_linked_module in [node for node, _ in value_list]:
                    if target_module not in [node for node, _ in value_list]:
                        index = [node for node, _ in value_list].index(best_linked_module)
                        value_list[index] = (target_module, value_list[index][1])
                    else:
                        value_list = [(node, weight) for node, weight in value_list if node != best_linked_module]
                current_possible_links[key] = value_list
        else:
            if verbose:
                print(f"No suitable module found to merge with {target_module}\n")

    if verbose:                   
        print()         
        print(f"{best_linked_module} have been merged to {target_module}\n") 
    
    return best_modules, new_possible_links


# 5. ITERATIVE INFOMAP ALGORITHM
#-----------------------------------------------------------------------------------------------------------------------------------

def iterative_infomap_partitioning (modules, possible_links, adjacent_matrix, 
                                    pagerank_on=True, 
                                    d=0.85, 
                                    tol=1e-10, 
                                    shuffle=False, 
                                    normalize_weights=True, 
                                    normalize_weights_after_merging=False,
                                    move_out=False,
                                    verbose=False): 
    
    """
    function that applies iterativly the infomap partition algorithm

        inputs: - modules: dictioray,
                - possible_links: dictionary,
                - adjacent_matrix: matrix,
                - pagerank_on: bool,
                - d: float,
                - tol: float,
                - normalize_weights: bool,
                - normalize_weihts_after_merging: bool,
                - verbose: bool

    outputs:    - best_modules: dictionary,
                - final_links: dictonary,
                - final_partition: dictionary,
    """

    graph_df = graph_info(adjacent_matrix, pagerank_on, d, tol, normalize_weights)

    if modules is None:
        modules = initialize_modules(graph_df)

    current_modules = copy.deepcopy(modules)
    modules_list = list(modules.keys())

    if shuffle:  
        if verbose:
            print(f"Before shuffling: {modules_list}")
        random.shuffle(modules_list)
        if verbose:
            print(f"After shuffling: {modules_list}\n")

    for id in modules_list:
        if id in current_modules.keys():

            current_possible_links = copy.deepcopy(possible_links)
            
            if possible_links is not None: 
                for i, j_list in current_possible_links.items(): 
                    if j_list is not None:
                        current_possible_links[i] = [j for j in j_list if j[0] != i]

            if verbose:
                print(f"current possible links in input: {current_possible_links}")
            
            current_modules, current_possible_links = merge_modules(current_modules, possible_links, id, graph_df, 
                                                                    pagerank_on=pagerank_on,
                                                                    d=d,
                                                                    normalize_weights_after_merging=normalize_weights_after_merging, 
                                                                    move_out=move_out,
                                                                    verbose=verbose)
            
            if current_possible_links is not None:
                possible_links = current_possible_links
                
            if verbose:
                print_modules(current_modules)
                print("-----------------------------------------------------------------------------------------------------------------------")

    new_modules_name = list(np.arange(len(current_modules)))
    old_modules_name = list(current_modules.keys())
    correspondance_list = []

    for i, j in enumerate(old_modules_name):
        correspondance_list.append(tuple([j, i]))

    final_modules = new_modules_name
    new_keys = new_modules_name 

    best_modules = {}
    for new_key, value in zip(new_keys, current_modules.values()):
        best_modules[new_key] = value

    if possible_links is not None:
        for i, j_list in possible_links.items():
            possible_links[i] = [j for j in j_list if j[0] != i]

        final_links = {}    
        for key, value in possible_links.items():
            new_key = next(new_key for old_key, new_key in correspondance_list if old_key == key)
            new_value = [(next(new_key for old_key, new_key in correspondance_list if old_key == v), w) for v, w in value]
            final_links[new_key] = sorted(new_value, key=lambda x: x[0])

    final_partition = {}
    for i in final_modules:
        final_partition[i] = [int(i) for i in np.sort(list(best_modules[i][0])[1:])]

    return best_modules, final_links, final_partition


# 6. VISUALIZE PARTITIONS
#-----------------------------------------------------------------------------------------------------------------------------------

def plot_graph_with_partitions(graph, partitions, 
                            layout="auto", 
                            edge_width=0.5, 
                            vertex_label=None, 
                            vertex_label_size=8.0,
                            vertex_size=25, 
                            plot_size=(7, 5), 
                            colour_map="nipy_spectral", 
                            title=""):
    """
    auxiliary function that plots the graph using different colors for nodes that are in different partitions.

    inputs:   - graph: igraph graph
                - partitions: dictionary, dictionary of partitioned nodes
                - layout: string,
                - edge_width: numeric,
                - vertex_label: list,
                - vertex_size: numeric
                - plot_size: tuple,
                - color_map: string, colours that will be used for the visualization
                - title: string, title of the plot

    output:   - plot of the graph with partitions
    """

    n = len(graph.vs)

    if vertex_label is None:       
        vertex_label = [str(i) for i in range(n)]

    vertex_colors = [None] * n

    cmap = plt.get_cmap(colour_map)
    num_colors = cmap.N

    for partition, nodes in partitions.items():
        color = cmap(partition % num_colors)  
        for node in nodes:
            vertex_colors[node] = color

    fig, ax = plt.subplots(figsize=plot_size)
    fig.tight_layout(pad=1.5)
    ig.plot(graph, layout=layout, 
            vertex_color=vertex_colors, 
            vertex_label=vertex_label, 
            vertex_label_size=vertex_label_size,
            vertex_size=vertex_size, 
            edge_width=edge_width, 
            target=ax)
    fig.suptitle(title, size=12)
    plt.show()