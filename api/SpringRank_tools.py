import networkx as nx
import numpy as np
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
        

       
def SpringRank_planted_network(N,beta,alpha,K,prng,l0=0.5,l1=1.):
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