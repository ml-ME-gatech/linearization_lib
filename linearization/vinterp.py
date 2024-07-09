import pandas as pd
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
import numpy as np
import argparse
import os
from typing import Tuple
import warnings

"""
Helper/Convinience classes for interpolating volume fields. This is different than interpolating 
to faces - interpolating to faces tends to b extremely difficult to do accurately, whereas 
interpplating in a volume is usually highly accururate if the mesh is "dense enough"
"""

def match_translate_bbox(x1: np.ndarray,
                         x2: np.ndarray,
                         tol = 1e-12) -> Tuple[np.ndarray, float]:
    

    dx = x1.min() - x2.min()
    x2 +=  dx
    scale = x1.max() - x1.min()
    diff = np.abs(x1.max() - x2.max())
    diff[diff < tol]  =1. 

    v_overlap = np.prod(diff/scale)

    return x2,v_overlap    

def interpolate_nodal_values(xin: np.ndarray,
                             yin: np.ndarray,
                             xout: np.ndarray) -> np.ndarray:

    """
    a light wrapping around the scipy's LinearNDInterpolater 
    + NearestNDInterpolator functions - tries to do linear interpolating
    and for points that failed (which occurs even if a point is barely outside
    a Delauny triangulation) do nearest neighbors interpolating

    Parameters
    ----------
    xin: np.ndarray
        the locations to interpolate at
    yin: np.ndarray 
        the values at the specified locations
    xout: np.ndarray 
        the locations to interpolate the provided values to

    Returns
    --------
    np.ndarray
        the interpolate values at xout
    
    """

    xin,v_overlap = match_translate_bbox(xout,xin)
    if yin.ndim == 1:
        yin = yin[:,None]
    
    linear_interp = LinearNDInterpolator(xin,
                                        yin,
                                        rescale = True,
                                        fill_value = np.nan)
                            
    values_out = linear_interp(xout)
    index = np.any(np.isnan(values_out),axis = 1)

    nearest_interp = NearestNDInterpolator(xin,
                                           yin,
                                           rescale = True)

    values_out[index,:] = nearest_interp(xout[index,:])

    return values_out,v_overlap

def interpolate_nodal_temperatures(df_in: pd.DataFrame,
                                   mesh_nodes: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolates scalar/vector values at supplied nodes in "df_in" 
    to the nodal locations supplied at "mesh_nodes". 

    Parameters
    ----------

    df_in : pd.DataFrame
        dataframe of input nodal temperatures, indexed on the the nodal numbers
        with the columns in order [x-coordinate,y-coordinate,z-coordinate,temperature]
        this will be supplied from an input cfd study
    
    mesh_nodes : pd.DataFrame
        datafame of mesh nodal locations, index on the nodal numbers with the columns
        ordered as [x-coordinate,y-coordinate,z-coordinate]. This will be supplied from
        a run-time or pre-supplied write of locations from ansys apdl
    """
    
    #perform linear interpolation for a majority of the points
    #filling values that are outside the range with nan
    xin = df_in[['{}-coordinate'.format(c) for c in ['x','y','z']]].to_numpy()
    yin = df_in['temperature'].to_numpy()
    xout = mesh_nodes.to_numpy()
    yout,v_overlap = interpolate_nodal_values(xin,yin,xout)
    return pd.DataFrame(yout,
                        index = pd.Series(mesh_nodes.index.astype(int),name = 'node'),
                        columns = ['temperature']),v_overlap


def main():

    #parsing logic here
    parser = argparse.ArgumentParser(description= 'Interpolation of temperatures using the scipy interpolate \
                                                    library. This function accepts three file names as arguments')
            
    parser.add_argument('file_name1',type = str,nargs = 1,
                        help = 'file containing cfd temperature data from fluent in \
                                node,x-coordinate,y-coordinate,z-coordinate,temperature .csv format')
    
    
    parser.add_argument('file_name2',type = str,nargs = 1,
                        help = 'file containing nodal locations from apdl in \
                                node,x-coordinate,y-coordinate,z-coordinate,.csv format')

    parser.add_argument('file_name3',type = str,nargs = 1,
                        help = 'file name to write interpolated temperature to \
                                in node,temperature .csv format')      


    args = parser.parse_args()

    #check to make sure the input files exist for sanity
    for arg in list(vars(args).values())[0:2]:
        if not os.path.exists(arg[0]):
            raise FileNotFoundError('file: {} does not exist, please provide an existing file name'.format(arg))
    
    #read in input data frames. pandas does not correctly format cfd dataframe headers
    cfd_df = pd.read_csv(args.file_name1[0],index_col = 0,header = 0,sep = ',')
    cfd_df.columns = [c.strip() for c in cfd_df.columns]
    nodal_df = pd.read_csv(args.file_name2[0],index_col = 0,header = None,sep = ',')

    #do the interpolation according to the logic supplied in the function here
    output_df,v_overlap = interpolate_nodal_temperatures(cfd_df,nodal_df)

    if v_overlap < 0.1:
        warnings.warn(f'The overlap between the cfd and nodal locations is: {round(v_overlap*100,3)}%')

    #write the interpolated dataframe to a file
    output_df.to_csv(args.file_name3[0])


if __name__ == '__main__':
    main() 


    