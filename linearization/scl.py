import numpy as np
import pandas as pd 
from typing import Tuple, List, Union

def _at_least2d(array: np.ndarray) -> np.ndarray:

    if array.ndim == 1:
        return array[:,None]
    else:
        return array

class SCL_APDLFile:

    """
    Class for reading in the stress classification line output from
    APDL - they have a pretty annoying format

    Parameters
    ----------
    file: str
        the name of the file 
    """

    ICO_HEADER = 'I=INSIDE C=CENTER O=OUTSIDE'
    NODE_HEADER = '***** POST1 LINEARIZED STRESS LISTING *****'
    RESULTS_HEADER = 'THE FOLLOWING X,Y,Z STRESSES'

    def __init__(self,file: str):

        self.file = file
        self.data = {}
        self.nodes = None

    def parse_nodes(self, text: str) -> None:

        lines = text.strip().splitlines()
        nds = []
        for line in lines:
            if 'INSIDE NODE' in line:
                chunks = line.strip().split()
                for chunk in chunks:
                    try:
                        n = int(chunk.strip())
                        nds.append(n)
                    except ValueError:
                        pass
                
                break
        
        self.nodes = nds

    def read(self):
        """
        read the APDL SCL File
        """
        with open(self.file,'r') as file:
            text = file.read()
        
        header,body = text.split(self.RESULTS_HEADER)

        self.parse_nodes(header)

        lines = []
        for i,line in enumerate(body.strip().splitlines()):
            if line.strip() == '' and i > 0:
                key,value = self.parse_chunk(lines)
                self.data[key] = value
                lines = []
            else:
                lines.append(line)
        
        key,value = self.parse_chunk(lines)
        self.data[key] = value

    def __getitem__(self,key: str) -> Union[pd.Series,pd.DataFrame]:
        return self.data[key]
    
    def parse_chunk(self,lines: str):

        if self.ICO_HEADER in lines[0]:
            return self._parse_ico_chunk(lines)
        else:
            return self._parse_non_ico_chunk(lines)
    
    def _parse_header(self,text) -> str:

        return ''.join(text.strip().split('**')[1:-1]).strip()

    def _parse_ico_chunk(self,lines: List[str]) -> Tuple[str,pd.DataFrame]:

        title = self._parse_header(lines[0])
        data = {}
        for i,line in enumerate(lines[1:]):
            if i % 4 == 0:
                keys = [k.strip() for k in line.strip().split()]
                for k in keys:
                    data[k] = []
            else:
                values = [float(v.strip()) for v in line.strip().split()[1:]]
                for k,v in zip(keys,values):
                    data[k].append(v)
        
        
        return title,pd.DataFrame.from_dict(data,orient= 'index',columns= ['I','C','O'])    

    def _parse_non_ico_chunk(self,lines: List[str]) -> Tuple[str,pd.Series]:

        title = self._parse_header(lines[0])
        data = {}
        keys = []
        values = []
        for i,line in enumerate(lines[1:]):
            if i % 2 == 0:
                keys = [k.strip() for k in line.strip().split()]
            else:
                values = [float(v.strip()) for v in line.strip().split()]
                for k,v in zip(keys,values):
                    data[k] = v
        
                keys = []
                values = []
        
        return title,pd.Series(data)


class SCL:
    """
    SCL
    makes stress classication points between pairs of nodes listed in x1 and x2
    usign the __call__ method

    npoints are placed between each node in x1 and x2 and the result is returned either
    as a nxdxp array where n is the number of nodes, and p is the number of integration points, and d is the dimension

    or if flattened output is requested it is returned as a (n*p)xd array
    """

    def __init__(self,  
                 x1: np.ndarray,
                 x2: np.ndarray):

        self.x1 = _at_least2d(x1)
        self.x2 = _at_least2d(x2)
        self.scl = None
    
    def _make_scl(self,npoints: int):
        vec = self.x2 - self.x1
        dim3 = np.linspace(0,1,npoints)

        self.scl = np.multiply.outer(vec,dim3).reshape(vec.shape[0],vec.shape[1],npoints)\
          + self.x1[...,None]

        self.scl = self.scl.swapaxes(-1,-2)
        
    
    def __call__(self,npoints: int,
                     flattened = False) -> np.ndarray:

        if self.scl is None:
            self._make_scl(npoints)
        
        if flattened:
            return self.scl.swapaxes(-1,2).\
                    reshape(self.scl.shape[0]*self.scl.shape[1],self.scl.shape[2])
        else:
            return self.scl

def main():

    """
    node_pairs = np.load('paired.npy')
    node_loc = pd.read_csv('node_locations.csv',index_col = 0,header= 0)

    loc1 = node_loc.loc[node_pairs[:,0]].to_numpy()
    loc2 = node_loc.loc[node_pairs[:,1]].to_numpy()

    scl = SCL(loc1,loc2)
    nodes = scl(47,flattened= True)
    print(nodes.flatten().shape[0]*1e-6)
    df = pd.DataFrame(nodes,columns = [1,2,3],index = pd.Series(np.arange(1,nodes.shape[0] + 1),name = 1))
    df.to_csv('scl.loc',sep = ',')
    """

    scl_file = SCL_APDLFile('beam_verification_lines/ls_scl1.txt')
    scl_file.read()
    print(scl_file.data['TOTAL'])

if __name__ == '__main__':
    main()