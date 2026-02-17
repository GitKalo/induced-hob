###
# HEADER!! 
###

from itertools import combinations

import numpy as np
rng = np.random.default_rng()

####################
# Network generation
####################

def make_hypergraph_simplicial(hg) :
    """
    Modifies a hypergraph in place to include all possible lower-order hyperedges 
    for each existing hyperedge.
    """
    for size in range(3, hg.max_size()+1) :
        top_edges = hg.get_edges(size=size)
        for e in top_edges :
            for sub_size in range(2, size) :
                # hg.add_edges(combinations(e, sub_size))
                for edge in combinations(e, sub_size) :
                    if not hg.check_edge(edge) :
                        hg.add_edge(edge)

######################
# Sampling node groups
######################

def is_clique_hgx(hg, nodes) :
    """
    Check if `nodes` forms a pairwise clique in the hgx.Hypergraph `hg`.
    """
    for u, v in combinations(nodes, 2) :
        if not hg.check_edge((u,v)) :
            return False
    return True

def cliques_of_node_hgx(hg, n, minsize=3, maxsize=3) :
    """
    Get all cliques of given sizes that node n participates in.

    G is a hypergraphx.Hypergraph object.
    """
    cliques = []
    for size in range(minsize, maxsize+1) :
        for nbrs in combinations(hg.get_neighbors(n, size=2), size-1) :
            if is_clique_hgx(hg, nbrs) :
                cliques.append(tuple(sorted([n, *nbrs])))

    return sorted(cliques)

def get_cliques_hgx(hg, group_size, n_groups=100) :
    """
    Sample `n_groups` pairwise cliques of size `group_size` that are 
    not hyperedges of size 3 or more from the hypergraphx.Hypergraph `hg`.

    If `n_groups` is None, return all cliques of the given size.
    If there are fewer cliques in the network than requested, return all available cliques.
    """
    N = hg.num_nodes()

    # Get all cliques of given size that are not hyperedges
    cliques_all = set()     # Use set to avoid duplicates
    for n in range(N) :
        for c in cliques_of_node_hgx(hg, n, minsize=group_size, maxsize=group_size) :
            if group_size <= 2 or not hg.check_edge(c) :    # Ensure not hyperedges, but allow up to size 2 (single nodes and pairwise edges)
                cliques_all.add(tuple(sorted(c)))
    
    if n_groups is None :
        # Return all cliques
        cliques = sorted(cliques_all)
    elif len(cliques_all) < n_groups :
        # Not enough cliques to sample from
        print(f"Warning: requested {n_groups} cliques but only {len(cliques_all)} exist. Returning all available cliques.")
        cliques = sorted(cliques_all)
    else :
        # Sample from all cliques without replacement
        cliques_all = list(cliques_all)   # Convert to list for sampling
        c_ids = rng.choice(len(cliques_all), n_groups, replace=False)
        cliques = sorted([cliques_all[i] for i in c_ids])
    
    return cliques

def get_random_groups_hgx(hg, group_size=3, n_groups=100) :
    """
    Sample a total of `n_groups` groups of random nodes of size `group_size` 
    uniformly at random from the hgx.Hypergraph `hg`.
    """
    # Get list of nodes
    nodes = hg.get_nodes()

    # Sample until we get `n_groups` viable groups
    random_groups = set()
    while len(random_groups) < n_groups :
        group = tuple(rng.choice(nodes, group_size, replace=False).tolist())    # list > tuple to avoid np types
        if \
        (group not in random_groups) and \
        (not hg.check_edge(group)) and \
        (not is_clique_hgx(hg, group))  :
            random_groups.add(tuple(sorted(group)))
    
    return sorted(random_groups)

#####################
# Dynamics simulaiton
#####################

def run_sis_sync_hgx(hg, betas, mu=1, t_max=100, init=0.1, rng=None) :
    """
    Discrete-time, synchronous update, probability-based SIS on HO networks with `hypergraphx` package.

    hg is a hypergraphx.Hypergraph object
    betas is a dictionary of beta values, with keys being the hyperedge **sizes**

    In a real application, we might use a different data structure for the results to save space 
    (e.g. a list of tuples, where each tuple holds a time index and a list of infected nodes)
    """
    # If no rng provided, create a fresh one.
    # Useful for working with multiple processes when this is the only 
    # function using random numbers.
    if rng is None :
        rng = np.random.default_rng()

    # Prevent beta > 1 (as it affects p_inf below, but could occur by accident)
    betas = {size : min(1, beta) for size, beta in betas.items()}

    N = hg.num_nodes()      # ONLY WORKS WHEN NODE INDICES ARE CONSECUTIVE
    nodes = hg.get_nodes()  # e.g. breaks if we took GC of disconnected graph
    
    # res = []
    res = np.zeros((t_max, N), dtype='int8')

    state = np.zeros(N)
    if type(init) is float :
        inf_init = []
        for n in nodes :
            if rng.random() < init :
                inf_init.append(n)
    elif type(init) is list :
        inf_init = init
    elif type(init) is np.ndarray :
        inf_init = init.tolist()
    else :
        raise ValueError("Unrecognized type for `init` parameter: should be either a fraction (float) or a list/array of nodes.")
    state[inf_init] = 1
    # res.append((0, np.where(state==1)[0].tolist()))
    res[0] = state

    t = 1
    n_i = np.sum(state)
    while t < t_max :
        state_new = state.copy()
        # Update each node based on previous time state
        for n_target in nodes :
            if state[n_target] == 1 :   # Recovery, independent
                if rng.random() < mu and n_i > 1 :  # Prevent steady state
                    state_new[n_target] = 0
                    n_i -= 1
            else :      # Infection, from neighbours and hyperedges of all orders (incl. pairwise)
                for hyperedge in hg.get_incident_edges(n_target) :
                    if sum(state[list(hyperedge)]) == len(hyperedge) - 1 and rng.random() < betas[len(hyperedge)] :
                    # if state[list(set(hyperedge) - {n_target})].all() and rng.random() < betas[len(hyperedge)] :
                        state_new[n_target] = 1
                        n_i += 1
                        break   # Avoid unnecessary checks if target gets infected

        state = state_new

        # res.append((t, np.where(state==1)[0].tolist()))
        res[t] = state

        # Increment and repeat
        t += 1

    return res

######################
# Information measures
######################

def get_p_joint_np(Xs, bins=2) :
    """
    Xs assumed to have time in axis 0.
    """
    vals, _ = np.histogramdd(Xs, bins=bins)
    p = vals + 10e-30   # Avoid division by zero downstream
    p /= np.sum(p)      # Normalize
    return p

def get_p_joint_all(X_all) :
    """
    Get joint probability distribution of all nodes and their one-step history.

    Output indices 0 to n-1 correspond to "current" state, n to 2n-1 to "previous" state.

    inf   (np.ndarray) : Array of node state with time along axis 0 and nodes along axis 1.
    nodes (list)       : Node indeces. Should be a list to allow propoer indexing 
                         of inf array.
    """   
    xs = X_all[1:]
    xs_history = X_all[:-1]
    xs_all = np.concatenate((xs, xs_history), axis=1)
    p_joint = get_p_joint_np(xs_all, bins=2)
    return p_joint

def cmi(p_yxsy0, n) :
    """
    Conditional mutual information between a single target and a set of source variables.
    Assumes that:
        - dim 0 of dist corresponds to target (y);
        - dims 1 to ndim-m correspond to sources (xs); and
        - last m dims correspond to target history (y0).
    """
    nbins = p_yxsy0.shape[0]     # To make more general (applicable to any hist distribution)
    
    p_y0 = np.sum(p_yxsy0, axis=tuple(range(0, n+1)))
    p_yy0 = np.sum(p_yxsy0, axis=tuple(range(1, n+1)))
    p_cond_y0 = p_yy0 / np.sum(p_yy0, axis=0)[None,:] # Add "empty" axis, similar to keepdims but controlled
    # Can also do this via the following:
    # print(p_yy0 / np.stack([np.sum(p_yy0, axis=0)]*nbins, axis=0))
    # print(p_yy0 / np.sum(p_yy0, axis=0, keepdims=True))  # Keepdims is fine if you operate along same axis

    h_cond_y0 = -np.sum(np.stack([p_y0]*nbins, axis=0) * p_cond_y0 * np.log2(p_cond_y0))

    p_xsy0 = np.sum(p_yxsy0, axis=0)
    p_cond_xsy0 = p_yxsy0 / np.sum(p_yxsy0, axis=0)[None,...]
    
    h_cond_xsy0 = -np.sum(p_xsy0[None,...] * p_cond_xsy0 * np.log2(p_cond_xsy0))
    
    mi = float(h_cond_y0 - h_cond_xsy0)
    
    return mi

def dO(p_joint, m=1, return_cmi=False) :
    """
    Assumes that:
        - dim 0 of p_joint corresponds to target;
        - dims 1 to ndim-m correspond to sources; and
        - last m dims correspond to target history.
    """
    n = p_joint.ndim-1-m    # Number of sources

    mi_yxn = cmi(p_joint, n)
    mi_yxj = [cmi(np.sum(p_joint, axis=j+1), n-1) for j in range(n)]
    
    dOn = float( (1-n)*mi_yxn + np.sum(mi_yxj) )

    if return_cmi :
        return dOn, mi_yxn, mi_yxj
    else :
        return dOn
    
def redundancy(p_joint, return_pw_mis=False) :
    """
    Expect target in axis 0, sources in axes 1 to len(sources), and target history in last axis.

    Works for single-target, single-step history.

    TODO: Make for more than one-step target history (arbitrary m)
    """
    ndim = p_joint.ndim
    s_ids = list(range(1, ndim-1))
    # Marginal MIs
    mis_pw = []
    for s in s_ids :
        # Marginalize over other source histories
        p = np.sum(p_joint, axis=tuple([s_id for s_id in s_ids if s_id != s]))
        mis_pw.append(cmi(p, 1))
    
    if return_pw_mis :
        return min(mis_pw), mis_pw
    else :
        return min(mis_pw)

def synergy(p_joint, mi_all=None, return_mi=False) :
    """
    Expect target in axis 0, sources in axes 1 to len(sources), and target history in last axis.
    """
    ndim = p_joint.ndim
    s_ids = list(range(1, ndim-1))
    if mi_all is None :
        mi_all = cmi(p_joint, len(s_ids))
    # One-out MIs
    mis_oo = []
    for s in s_ids :
        # Marginalize over source history
        p = np.sum(p_joint, axis=s)
        mis_oo.append(cmi(p, len(s_ids)-1))
    
    if return_mi :
        return mi_all - max(mis_oo), mi_all
    else :
        return mi_all - max(mis_oo)
    
######################
# Statistical distance
######################

def get_stat_dist_for_index(i, measure, mask_g1, mask_g2, n_bins=20) :
    """
    Get the statistical distance ("delta") between the distributions of values given by
    the `measure` parameter at index `i` for the two groups defined by
    array masks `mask_g1` and `mask_g2`.
    """
    # Distance needs to be over common alphabet (specific to each i)
    measure = np.nan_to_num(measure)  # Convert NaNs to 0s for computing distance
    bin_min = min(np.min(measure[i][mask_g1]), np.min(measure[i][mask_g2]))
    bin_max = max(np.max(measure[i][mask_g1]), np.max(measure[i][mask_g2]))
    common_bin_range = (bin_min, bin_max)

    # Get probability distributions
    dist_g1, _ = np.histogram(measure[i][mask_g1], bins=n_bins, range=common_bin_range)
    dist_g2, _  = np.histogram(measure[i][mask_g2],  bins=n_bins, range=common_bin_range)
    dist_g1 = dist_g1.astype(float) / np.sum(dist_g1)
    dist_g2 = dist_g2.astype(float) / np.sum(dist_g2)

    # Calculate statistical distance
    return np.sum(np.abs(dist_g2 - dist_g1))/2