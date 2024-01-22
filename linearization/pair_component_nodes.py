import pandas as pd
import numpy as np
from typing import List, Union, Set
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

class Reindexer:

    """
    reindexes a node list so that for a given node list with n unique
    identifiers, the nodes are listed as 0,..,n-1. This is convinient 
    for computations on the graph where row/col position denotes node number

    Parameters
    ----------
    n1: List[int]
        the integer values of the first nodes
    n2: List[int]
        the integer values of the second nodes 
    """

    def __init__(self,n1: List[int],
                      n2: List[int]):

        self.new_nodes = n1
        self.old_nodes = n2

        self.__forward = dict((zip(n1,n2)))
        self.__backward = dict((zip(n2,n1)))
    
    @property
    def forward(self) -> dict:
        return self.__forward
    
    @property
    def backward(self) -> dict:
        return self.__backward
    
    def _transform(self,array: Union[Set,List,np.ndarray],
                               direction: str) -> np.ndarray:

        _array = np.array(array)

        return np.array([self.__getattribute__(direction)[a] 
                        for a in _array.flatten()]).reshape(_array.shape)
                    
    def forward_transform(self,array: Union[Set,List,np.ndarray]) -> np.ndarray:
        
        return self._transform(array,'forward')
    
    def backward_transform(self,array:Union[Set,List,np.ndarray]) -> np.ndarray:
        
        return self._transform(array,'backward')

    @classmethod
    def from_node_list(cls,node_list: List[int],
                            assume_sorted = False):

        if not assume_sorted:
            node_list.sort()
    
        nl = list(np.arange(0,len(node_list),1,dtype = int)) 
        return cls(node_list,nl)

class LSANodePairer:
    """
    class for pairing nodes between two nodes lists using the "linear sum assignment"
    problem. That is, given two bipartite graphs, U and V, with distances between nodes in each graph
    (i.e. let v be a node in graph V, then for u_i in graph U there are distances defined 
    between v and u_i for at least one i), in this case for all nodes in the graph, 
    we seek the paring between the graphs so that each node in graph U has exactly one
    pairing with a node in graph V. 

    This works for graphs of different sizes, it will just find the pairing between nodes for the smaller
    graph
    """

    def __init__(self,nodes1: Union[np.ndarray, pd.DataFrame], 
                      nodes2: Union[np.ndarray, pd.DataFrame],
                      locations: Union[pd.DataFrame, None]):

        if nodes1.shape[0] <= nodes2.shape[0]:
            self.nodes1 = nodes1 
            self.nodes2 = nodes2
        else:
            self.nodes1 = nodes2
            self.nodes2 = nodes1
        
        if locations is None:
            self.locations = pd.concat([self.nodes1,self.nodes2],axis = 0)
            self.nodes1 = self.nodes1.index
            self.nodes2 = self.nodes2.index 
            nodes = np.unique(np.concatenate([nodes1.index.to_numpy(),
                               nodes2.index.to_numpy()],axis = 0))
            self.reindexer = Reindexer.from_node_list(nodes)
        else:
            self.locations = locations
            self.reindexer = Reindexer.from_node_list(
                                        np.unique(locations.index.to_numpy())
                                        )
        
        self.__distance_matrix = None


    @property
    def r_nodes1(self) -> np.ndarray:
        return self.reindexer.forward_transform(self.nodes1)
    
    @property
    def r_nodes2(self) -> np.ndarray:
        return self.reindexer.forward_transform(self.nodes2)

    @property
    def r_locations(self) -> pd.DataFrame:
        return pd.DataFrame(self.locations.to_numpy(),
                            index = self.reindexer.forward_transform(self.locations.index))           

    def pair(self,
             num_closest = None,
             reindex = False) -> np.ndarray:

        """
        pairs the two graphs
        """
        if num_closest is None:
            paired = self._pair_full()
        else:
            paired = self._pair_sparse(num_closest)
        
        if reindex:
            return paired
        else:
            return self.reindexer.backward_transform(paired)
    
    @property
    def distance_matrix(self):
        if self.__distance_matrix is None:
            self.__distance_matrix = cdist(
                        self.r_locations.loc[self.r_nodes1].to_numpy(),
                        self.r_locations.loc[self.r_nodes2].to_numpy()
                        )
        
        return self.__distance_matrix
    
    @classmethod
    def from_locations(cls,nodes1: pd.DataFrame,
                           nodes2: pd.DataFrame):

        return cls(nodes1,nodes2,None)
    
    def _pair_sparse(self,num_closest: int) -> np.ndarray:

        raise NotImplementedError('havent implmented this yet, may be neccessary for very large problems?')
        

    def _pair_full(self) -> np.ndarray:
        """
        calls the linear sum assignment function from 
        scipy's library and returns the paired nodes
        """
        matched = linear_sum_assignment(self.distance_matrix)
        return np.array([self.r_nodes1[matched[0]], 
                        self.r_nodes2[matched[1]]]).T



