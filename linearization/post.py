from vinterp import interpolate_nodal_values
from linearization import APDLIntegrate
from scl import SCL
from pair_component_nodes import LSANodePairer
import pandas as pd
import numpy as np
import os
import pathlib
import sys
from typing import Tuple
import argparse
import shutil

"""
system determining stuff for working with pathlib on unix or windows
as this script is used in both environements
"""

platform =  str(sys.platform)

if platform == 'unix' or platform == 'posix' or platform == 'linux':
    _PATH = pathlib.PosixPath
else:
    _PATH = pathlib.WindowsPath

"""
Author: Michael Lanahan
Date Created: 10.12.2022
Last Edit: 04.29.2023

"""

def pair_nodes(parent: pathlib.PurePath) -> Tuple[pd.DataFrame,pd.DataFrame]:
    """ 
    pair the nodes between the two boundaries and save the pairing "map" that maps
    the nodes (indices on the dataframes) between the two sets of nodes

    Parameters
    ----------
    parent: pathlib.PurePath
        a pathlib object folder, the parent directory where the save the results

    Returns
    ----------
    Tuple[pd.DataFrame,pd.Dataframe]
        the node locations sorted in order according to the determined mapping
    
    """
    
    top_surface_nodes = pd.read_csv('ts.node.loc',index_col = 0,header = None)
    top_surface_nodes.index = top_surface_nodes.index.astype(int)

    bottom_surface_nodes = pd.read_csv('bs.node.loc',index_col = 0,header = None)
    bottom_surface_nodes.index = bottom_surface_nodes.index.astype(int)

    #pair nodes using linear sum assignment algorithm from
    #scipy
    node_pair = LSANodePairer.from_locations(top_surface_nodes,
                                            bottom_surface_nodes)
    
    paired_loc = node_pair.pair()

    #save the node locations to a file and return the paired nodes as
    #two numpy arrays where the integer value in each array is the node
    #on the surface paired between the two surfaces
    loc1 = top_surface_nodes.loc[paired_loc[:,0]]
    loc2 = bottom_surface_nodes.loc[paired_loc[:,1]]
    np.save(str(parent.joinpath('paired.npy')),paired_loc)

    return loc1,loc2

def linearize_stresses(parent: pathlib.PurePath,
                        loc1: pd.DataFrame,
                        loc2: pd.DataFrame,
                        npoints: int) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:

    """
    Linearize stress fields according to the guidelines provided in ASME
    code/ITER SDC document

    Parameters
    ----------
    parent: pathlib.PurePath
        a pathlib object folder, the parent directory where the save the results
    
    loc1: pd.DataFrame
        node locations of the first boundary sorted according to the determined mapping
    
    loc2: pd.DataFrame
        node locations of the second bounary sorted according to the determined mapping
    
    npoints: int
        the number of integration points to interpolate the stress tresults to

    Returns
    -------
    Tuple[membrane: np.ndarray,
          bending: np.ndarray,
          peak: np.ndarray]
        
    membrane,bending, and peak are the membrane,bending and peak bending stresses
    at all intermediate poitns on the plane between the two boundaries
    """

    scl_apdl = SCL(loc1.to_numpy(),loc2.to_numpy())
    scl_points = scl_apdl(npoints,flattened = True)

    node_loc = pd.read_csv('pb.node.loc',index_col = 0,header = None)
    node_sol = pd.read_csv('pb.node.dat',index_col = 0,header = None)

    node_loc = node_loc.loc[node_sol.index]

    scl_sol = interpolate_nodal_values(node_loc.to_numpy(),
                                       node_sol.to_numpy(),
                                       scl_points)

    apdl_int = APDLIntegrate(scl_sol,scl_points,npoints)
    membrane = apdl_int.membrane_vm(averaged = True)
    bending = apdl_int.bending_vm(averaged = True)
    peak = apdl_int.peak_vm(averaged = True)
    principal = apdl_int.linearized_principal_stress(averaged = True)
    triaxility_factor = apdl_int.triaxiality_factor(averaged = True)
    location = (loc1.to_numpy() + loc2.to_numpy())/2

    for a,name in zip([membrane,bending,peak,location,principal,triaxility_factor],\
                     ['membrane','bending','peak','location','principal','triaxility_factor']):

        np.save(str(parent.joinpath(name + '.npy')),a)
    
    return membrane,bending,peak

def thickness_average_temperature(parent: pathlib.PurePath,
                                    loc1: pd.DataFrame,
                                    loc2: pd.DataFrame,
                                    npoints: int) -> np.ndarray:

    """
    thickness average temperature fields according to the guidelines provided in ASME
    code/ITER SDC document. The logic here is almost identical to the 
    logic in linearize_stresses, except only one (scalar) value is "linearized" 
    or averaged across the thickness

    Parameters
    ----------
    parent: pathlib.PurePath
        a pathlib object folder, the parent directory where the save the results
    
    loc1: pd.DataFrame
        node locations of the first boundary sorted according to the determined mapping
    
    loc2: pd.DataFrame
        node locations of the second bounary sorted according to the determined mapping
    
    npoints: int
        the number of integration points to interpolate the stress tresults to

    Returns
    -------
    ta_temp: np.ndarray
        an array of temperatures averaged across the thickness and positioned
        at the intermediate plane between the two boundaries
    """
    
    scl_apdl = SCL(loc1.to_numpy(),loc2.to_numpy())
    scl_points = scl_apdl(npoints,flattened = True)

    node_loc = pd.read_csv('pb.node.loc',index_col = 0,header = None)
    node_temp = pd.read_csv('pb.node.cfdtemp',index_col = 0,header = None)

    node_loc = node_loc.loc[node_temp.index]

    scl_temp = interpolate_nodal_values(node_loc.to_numpy(),
                                       node_temp.to_numpy()[:,-1],
                                       scl_points)

    apdl_int = APDLIntegrate(scl_temp,scl_points,npoints)
    ta_temp = apdl_int.thickness_average()

    np.save(str(parent.joinpath('thickness_averaged_temperature.npy')),ta_temp)

    return ta_temp

def parse_args():
    #parsing logic here
    parser = argparse.ArgumentParser(description= 'command line post processing of linearized stresses')
            
    parser.add_argument('ea_type',type = str,nargs = 1,
                        help = 'the type of elastic analysis that this post-processor is supposed to consider')
    
    
    parser.add_argument('-n',type = int,nargs = 1,default = 47,
                         help = 'the number of points to integrate along: default 47 (in accordance with APDL builtin)')
    
    parser.add_argument('-p',type = str,nargs = 1,default = None,
                        help = 'path to write to')

    args = parser.parse_args()
    return args

def main():

    args = parse_args()
    
    ea_type = args.ea_type[0].lower()

    if ea_type != 'primary' and ea_type != 'secondary':
        raise ValueError('elastic analysis type (ea_type) must be one of (1) primary or (2) secondary')

    try:
        npoints = args.n[0]
    except TypeError:
        npoints = args.n

    pwd = args.p
    if pwd is None:
        pwd = os.path.split(__file__)[0]
    
    parent = _PATH(pwd).joinpath(ea_type)
    
    if parent.exists():
        shutil.rmtree(str(parent))
    
    os.mkdir(ea_type)
    
    loc1,loc2 = pair_nodes(parent)
    membrane,bending,peak = linearize_stresses(parent,loc1,loc2,npoints)

    if ea_type == 'secondary':
        ta_temp = thickness_average_temperature(parent,loc1,loc2,npoints)



if __name__ == '__main__':
    main()
