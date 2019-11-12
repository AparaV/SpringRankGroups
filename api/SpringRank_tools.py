import networkx as nx
import numpy as np
from scipy import sparse as sp
from scipy.sparse import coo_matrix,spdiags,csr_matrix
import scipy.sparse.linalg

import warnings
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)

def csr_SpringRank(A):
    """
    Main routine to calculate SpringRank by solving linear system
    Default parameters are initialized as in the standard SpringRank model
    
    INPUT:

        A=network adjacency matrix (can be weighted)
        alpha: controls the impact of the regularization term
        l0: regularization spring's rest length
        l1: interaction springs' rest length
        solver: linear system solver. Options: 'spsolve'(direct, slower) or 'bicgstab' (iterative, faster)
        verbose: if True, then outputs some info about the numerical solvers

    OUTPUT:
        
        rank: N-dim array, indeces represent the nodes' indices used in ordering the matrix A

    """
    N = A.shape[0]
    k_in = A.sum(axis=0)
    k_out = A.sum(axis=1).transpose()
    # form the graph laplacian
    operator = csr_matrix(
        spdiags(k_out+k_in,0,N,N)-A-A.transpose()
        )
    # form the operator A (from Ax=b notation)
    # note that this is the operator in the paper, but augmented
    # to solve a Lagrange multiplier problem that provides the constrain
    operator.resize((N+1,N+1))
    # print(operator.shape)
    operator[N,0] = 1
    operator[0,N] = 1

    # form the solution vector b (from Ax=b notation)
    solution_vector = np.append((k_out-k_in),np.matrix(0),axis=1).transpose()

    # perform the computations
    ranks = scipy.sparse.linalg.bicgstab(
        scipy.sparse.csr_matrix(operator),
        solution_vector
        )[0]

    return ranks


def SpringRank(A,alpha=0.,l0=1.0,l1=1.0,solver='bicgstab',verbose=False):
    N = A.shape[0]
    k_in = np.sum(A, 0)
    k_out = np.sum(A, 1)

    C= A+A.T
    D1 = np.diag(k_out + k_in)
    d2 = k_out - k_in

    if alpha!=0.: 
        if verbose==True:print('Using alpha!=0: matrix is invertible')

        B = alpha*l0 + d2
        A = alpha*np.eye(N)+ D1 - C
        A = scipy.sparse.csr_matrix(np.matrix(A))

        if solver=='spsolve':
            if verbose==True:print('Using scipy.sparse.linalg.spsolve(A,B)')
            rank = scipy.sparse.linalg.spsolve(A,B)
            # rank=np.linalg.solve(A,B)  # cannot use it with sparse matrices
            return np.transpose(rank)
        elif solver=='bicgstab': 
            if verbose==True:print('Using scipy.sparse.linalg.bicgstab(A,B)')
            rank=scipy.sparse.linalg.bicgstab(A,B)[0]
            return np.transpose(rank)
        else:
            print('Using scipy.sparse.linalg.bicgstab(A,B)')
            rank=scipy.sparse.linalg.bicgstab(A,B)[0]   

    else:    
        if verbose==True:print('alpha=0, using faster computation: fixing a rank degree of freedom')
        
        C= C+np.repeat(A[N-1,:][None],N,axis=0)+np.repeat(A[:,N-1].T[None],N,axis=0)
        d3 = np.full((N,), k_out[N-1] - k_in[N-1])
        B = d2 + d3
        # A=D1-C
        A=scipy.sparse.csr_matrix(np.matrix(D1-C))
        if solver=='spsolve':
            if verbose==True:print('Using scipy.sparse.linalg.spsolve(A,B)')
            rank = scipy.sparse.linalg.spsolve(A,B)
        elif solver=='bicgstab': 
            if verbose==True:print('Using scipy.sparse.linalg.bicgstab(A,B)')
            rank=scipy.sparse.linalg.bicgstab(A,B)[0]
        else:
            print('Using scipy.sparse.linalg.bicgstab(A,B)')
            rank=scipy.sparse.linalg.bicgstab(A,B)[0]
        return np.transpose(rank)


def SpringRank_groups(A, G, reg_coeff, solver):
    """
    Solve SpringRank with groups

    Arguments:
        A: The directed network (np.ndarray)
        G: Dictionary of the group assignment matrices
        reg_coeff: Dictionary of regularization coeffecients.
            Expects same keys as `G` and an additional "individual" key.
        solver: The sparse solver to be used

    Output:
        ranks: Final combined ranks (np.ndarray)
        scores: Dictionary of scores, sorted by groups.
            Has same keys as `G` and an additional "individual" key.
    """
    
    # Get array shapes
    N, M = A.shape
    assert(N == M)
    
    # Construct Laplacian
    k_in = np.sum(A, 0)
    k_out = np.sum(A, 1)
    D = np.diag(k_out + k_in)
    L = D - (A + A.T)
    
    # Make everything sparse
    L = csr_matrix(L)
    num_groups = {}
    G_sparse = {}
    for group_type, G_i in G.items():
        num_groups[group_type] = G_i.shape[1]
        G_sparse[group_type] = csr_matrix(G_i)
    
    # Construct the LHS matrix (sparse) and RHS vector (dense)
    blocks = {}
    for group_type, G_i in G.items():
        blocks[group_type] = L @ G_i
    
    K = L + reg_coeff["individual"] * sp.eye(N)
    for group_type in G:
        K = sp.hstack([K, blocks[group_type]])
    k_diff = k_out - k_in
    d_hat = k_diff
    
    for group_type, lambda_i in reg_coeff.items():
        if group_type == "individual":
            continue
        G_i_sparse = G_sparse[group_type]
        G_i = G[group_type]
        n_i = num_groups[group_type]
        current_row = G_i_sparse.T @ L
        for block_type, block in blocks.items():
            if block_type == group_type:
                current_row = sp.hstack([current_row, G_i_sparse.T @ block + lambda_i*sp.eye(n_i)])
            else:
                 current_row = sp.hstack([current_row, G_i_sparse.T @ block])
        K = sp.vstack([K, current_row])
        
        d_i = np.matmul(G_i.T, k_diff)
        d_hat = np.append(d_hat, d_i, axis=0)
    
    # Solve using sparse or iterative solvers
    if solver == 'spsolve':
        x = scipy.sparse.linalg.spsolve(K, d_hat)
    elif solver == 'bicgstab':
        output = scipy.sparse.linalg.bicgstab(K, d_hat)
        x = output[0]
    elif solver == 'lsqr':
        output = scipy.sparse.linalg.lsqr(K, d_hat)
        x = output[0]
    else:
        output = scipy.sparse.linalg.bicgstab(K, d_hat)
        x = output[0]
    
    # Make x dense
    try:
        x = x.toarray()
    except AttributeError:
        pass
    
    # Rearrange scores and compute ranks
    scores = {}
    scores["individual"] = x[:N]
    ranks = scores["individual"]
    prev_idx = N
    for group_type, n_i in num_groups.items():
        scores[group_type] = x[prev_idx:prev_idx + n_i]
        ranks += np.matmul(G[group_type], scores[group_type])
        prev_idx += n_i
    
    return ranks, scores

       
def SpringRank_planted_network(N, beta, alpha, K, prng, l0=0.5, l1=1.):
    '''

    Uses the SpringRank generative model to build a directed, possibly weigthed and having self-loops, network.
    Can be used to generate benchmarks for hierarchical networks

    Steps:
        1. Generates the scores (default is factorized Gaussian)
        2. Extracts A_ij entries (network edges) from Poisson distribution with average related to SpringRank energy

    INPUT:

        N=# of nodes
        beta= inverse temperature, controls noise
        alpha=controls prior's variance
        K=E/N  --> average degree, controls sparsity
        l0=prior spring's rest length 
        l1=interaction spring's rest lenght

    OUTPUT:
        G: nx.DiGraph()         Directed (possibly weighted graph, there can be self-loops)
        
    '''
    G=nx.DiGraph()

    scores=prng.normal(l0,1./np.sqrt(alpha*beta),N)  # planted scores ---> uses factorized Gaussian
    for i in range(N):G.add_node(i,score=scores[i])

    #  ---- Fixing sparsity i.e. the average degree  ---- 
    Z=0.
    for i in range(N):
        for j in range(N):  
            Z+=np.exp(-0.5*beta*np.power(scores[i]-scores[j]-l1,2))
    c=float(K*N)/Z        
    #  --------------------------------------------------

    # ----  Building the graph   ------------------------ 
    for i in range(N):
        for j in range(N):

            H_ij=0.5*np.power((scores[i]-scores[j]-l1),2)
            lambda_ij=c*np.exp(-beta*H_ij)

            A_ij=prng.poisson(lambda_ij,1)[0]

            if A_ij>0:G.add_edge(i,j,weight=A_ij)

    return G

def SpringRank_planted_network_groups(N, num_groups, beta, alpha, K, prng, l0=0.5, l0_g=0, l1=1,
                                      allow_self_loops=False, return_ranks=False):
    """
    Uses SpringRank generative model to build a directed, weighted network assuming group preferences.
    
    1. Randomly assign groups
    2. Generate scores assuming a normal distribution
    3. Generate network as described by the SpringRank generative model
    
    Arguments:
        N: Number of nodes
        num_groups: Dictionary of different group sizes
        beta: Inverse temperature
        alpha: Dictionary controlling individual and group scores' variance.
            Expects same keys as `num_groups` and an additional "individual" key.
        K: Average degree
        prng: Random number generator
        l0: Dictionary of individual and group scores' mean
            Expects same keys as `num_groups` and an additional "individual" key.
        l1: Spring rest length
        allow_self_loops: Allow self loops in network. Defaults to False
        return_ranks: Should we return the generated ranks. Defaults to False
    
    Output:
        A: nx.DiGraph()
        G: Dictionary of assigned group matrix
        scores: Dictionary of individual and group scores
        ranks: Generated total ranks
    """
    
    # Assign groups and generate scores

    scores = {}
    G = {}

    alpha_i = alpha["individual"]
    l0_i = l0["individual"]
    scores["individual"] = prng.normal(l0_i, 1/np.sqrt(alpha_i*beta), N)

    ranks = scores["individual"]

    for group_type in num_groups:

        # generate groups
        n_i = num_groups[group_type]
        groups_i = np.random.randint(0, n_i, N)
        G_i = np.zeros((N, n_i))
        for j, g_j in enumerate(groups_i):
            G_i[j, g_j] = 1
        G[group_type] = G_i
        
        # generate scores
        alpha_i = alpha[group_type]
        l0_i = l0[group_type]
        scores[group_type] = prng.normal(l0_i, 1/np.sqrt(alpha_i*beta), n_i)

        # compute rank
        ranks += np.matmul(G_i, scores[group_type])
    
    # Fix sparsity using the average degree
    scaled_exp_energy = np.zeros((N, N))
    Z = 0
    for i in range(N):
        for j in range(N):
            energy_ij = 0.5 * np.power(ranks[i]-ranks[j]-l1, 2)
            scaled_exp_energy[i, j] = np.exp(-beta * energy_ij)
            Z += scaled_exp_energy[i, j]
    c = float(K * N) / Z
    
    # Build network
    A = nx.DiGraph()
    for i in range(N):
        A.add_node(i, score=ranks[i])
    
    for i in range(N):
        for j in range(N):
            if i == j and not allow_self_loops:
                continue

            lambda_ij = c * scaled_exp_energy[i, j]
            A_ij = np.random.poisson(lambda_ij)
            if A_ij > 0:
                A.add_edge(i,j,weight=A_ij)
    
    if return_ranks:
        return A, G, scores, ranks
    else:
        return A, G     