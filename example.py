import scipy as sp
import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix

A = csr_matrix((4, 4), dtype=np.int8).toarray()
G = nx.from_scipy_sparse_matrix(A)

print("a: ", A)
print("g : ", G.graph)
print(type(G))
