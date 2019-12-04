import networkx as nx
import numpy as np

from scipy.optimize import brentq
from numba import jit

def build_graph_from_adjacency(inadjacency):
    """
    Takes an adjacency_list like: "23 41 18" or 18 times  "23 41 1"   (edge from 23 --> 41)
    possibly having multiple edges and build a graph with no multiple edges but weigths representing how many of them there are
    Necessary in case of using algorithms that do not accept MultiGraphs. E.g. eigenvector centrality.    
    """
    
    adjacency_list=open(inadjacency,'r')
    edges={}
    for row in adjacency_list:
        a=row.split()
        e=(a[0],a[1])
        w=int(a[2])
        if(e not in edges):edges[e]=w
        else:edges[e]+=w
    G=nx.DiGraph()
    for e in edges: G.add_edge(e[0],e[1],weight=edges[e])
    adjacency_list.close()

    return G

def btl(A,tol):
    N = np.shape(A)[0]
    g = np.random.rand(N)
    wins = np.array(np.sum(A,axis=1)).flatten();
    matches = np.array(A+np.transpose(A));
    totalMatches = np.array(np.sum(matches,axis=0)).flatten()
    g_prev = np.random.rand(N)
    eps = 1e-6
    while np.linalg.norm(g-g_prev) > tol:
        g_prev = g
        for i in range(N):
            if totalMatches[i]>0:
                q = np.divide(matches[i,:],g_prev[i]+g_prev)
                q[i] = 0
                g[i] = (wins[i]+eps)/np.sum(q)
            else:
                g[i] = 0
        g = g/np.sum(g)
    return np.log(g)

def adjust_ranks(ranks, matrix, least_rank=0, p_ij=0.8, interval=(0.01, 20)):
    """
    Apply linear transformation of ranks given p_ij and the least_rank
    """
    ranks = scale_ranks(ranks, matrix, p_ij, interval)
    ranks = shift_ranks(ranks, least_rank)
    return ranks

def scale_ranks(ranks, matrix, p_ij, interval):
    """
    Scale the ranks given p_ij
    """
    temperature = get_temperature(ranks, matrix, p_ij, interval)
    return ranks * temperature

def shift_ranks(ranks, least_rank=0):
    """
    Shifts all ranks so that the minimum is the least_rank
    """
    offset = np.min(ranks) - least_rank
    return ranks - offset

def get_temperature(ranks, matrix, p_ij, interval):
    """
    Calculate the correct scaling for ranks given p_ij
    """
    betahat = get_betahat(ranks, matrix, interval)
    log_odds = np.log(p_ij / (1 - p_ij))
    temperature = 2 * betahat / log_odds
    return temperature

def get_betahat(ranks, matrix, interval):
    """
    Solve Eq. 39 to find beta_hat using brentq solver from scipy.optimize
    """   
    return brentq(eqs39, interval[0], interval[1], args=(ranks, matrix))

@jit(nopython=True)
def eqs39(beta, s, A):
    N = A.shape[0]
    x = 0
    for i in range(N):
        for j in range(N):
            if A[i, j] == 0:
                continue
            else:
                x += (s[i] - s[j]) * (A[i, j] - (A[i, j] + A[j, i]) / (1 + np.exp(-2 * beta * (s[i] - s[j]))))
    return x