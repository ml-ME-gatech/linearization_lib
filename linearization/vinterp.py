import pandas as pd
from scipy.interpolate import LinearNDInterpolator,NearestNDInterpolator
import numpy as np

from typing import Tuple


"""
Helper/Convinience classes for interpolating volume fields. This is different than interpolating 
to faces - interpolating to faces tends to b extremely difficult to do accurately, whereas 
interpplating in a volume is usually highly accururate if the mesh is "dense enough"
"""

def match_translate_bbox(x1: np.ndarray,
                         x2: np.ndarray,
                         tol = 1e-20) -> Tuple[np.ndarray, float]:
    

    dx = x1.min(axis = 0) - x2.min(axis = 0)
    x2 +=  dx
    scale = x1.max(axis = 0) - x1.min(axis = 0)
    diff = np.abs(x1.max(axis = 0) - x2.max(axis = 0))
    diff[diff < tol]  =1. 

    v_diff = np.prod(diff/scale)

    return x2,1- v_diff    

def interpolate_nodal_values(xin: np.ndarray,
                             yin: np.ndarray,
                             xout: np.ndarray,
                             method = 'linear') -> np.ndarray:

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
    
    if method == 'linear':
        linear_interp = LinearNDInterpolator(xin,
                                            yin,
                                            rescale = True,
                                            fill_value = np.nan)
                            
        values_out = linear_interp(xout)
    elif method == 'nearest':
        values_out = np.nan*np.ones([xout.shape[0],yin.shape[1]])
    else:
        raise ValueError('method must be one of "linear" or "nearest"')
    
    index = np.any(np.isnan(values_out),axis = 1)

    nearest_interp = NearestNDInterpolator(xin,
                                           yin,
                                           rescale = True)

    values_out[index,:] = nearest_interp(xout[index,:])

    return values_out,v_overlap

def interpolate_nodal_temperatures(df_in: pd.DataFrame,
                                   mesh_nodes: pd.DataFrame,
                                   method = 'linear') -> pd.DataFrame:
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
    yout,v_overlap = interpolate_nodal_values(xin,yin,xout,method = method)
    return pd.DataFrame(yout,
                        index = pd.Series(mesh_nodes.index.astype(int),name = 'node'),
                        columns = ['temperature']),v_overlap



    