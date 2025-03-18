import numpy as np
from scipy.integrate import simpson,trapezoid

def _repeated_quantity(output: np.ndarray,
                       averaged: bool,
                       size: int) -> np.ndarray:

    if averaged:
        return output
    else:
        return np.repeat(output[...,np.newaxis],size,axis = -1)

def _linear_quantity(output: np.ndarray,
                     averaged: bool,
                     size: int) -> np.ndarray:

    if averaged:
        return output
    else:
        return np.linspace(output,-output,size).swapaxes(0,2).swapaxes(0,1)

def _von_mises_from_primary(stress: np.ndarray):
    """
    compute von-mises stress intensity from the primary criterion
    """
    return (0.5**0.5)*np.sqrt(np.power(stress[:,0,...] - stress[:,1,...],2.0) + \
                            np.power(stress[:,1,...] - stress[:,2,...],2.0) + \
                            np.power(stress[:,2,...] - stress[:,0,...],2.0))

def _von_mises_from_full(stress: np.ndarray):
    """
    compute von-mises stress intensity from the full (symmetric) stress criterion
    """
    return  np.sqrt(0.5*(np.power(stress[:,0,...] - stress[:,1,...],2.0) + \
                                np.power(stress[:,1,...] - stress[:,2,...],2.0) + \
                                np.power(stress[:,2,...] - stress[:,0,...],2.0) + \
                                6*(np.power(stress[:,3,...],2.0) + np.power(stress[:,4,...],2.0) +\
                                np.power(stress[:,5,...],2.0))))

def tresca_from_primary(sigma: np.ndarray):
    """
    compute tresca stress intensity from the primary criterion
    """

    perm_matrix = np.array([
        [ 1, -1,  0],  # sigma1 - sigma2
        [ 0,  1, -1],  # sigma2 - sigma3
        [-1,  0,  1]   # sigma3 - sigma1
    ])

    shear = np.einsum('ij,a...j->a...i', perm_matrix, sigma)
    np.abs(shear,out = shear)
    return shear.max(axis = 1) 
    
def tresca_from_full(sigma: np.ndarray):

    return tresca_from_primary(principal_values(sigma))

def equivalent_stress(stress: np.ndarray) -> np.ndarray:
    """
    compute equivalent stresses
    """
    if stress.shape[1] == 3:
        return _von_mises_from_primary(stress)
    else: 
        return _von_mises_from_full(stress)
    
def _equivalent_strain_from_full(strain: np.ndarray) -> np.ndarray:
    
    return 2.**0.5/3.*np.sqrt(
        (strain[:,0,...]*strain[:,1,...])**2 + \
        (strain[:,1,...] - strain[:,2,...])**2 + \
        (strain[:,2,...] - strain[:,0,...])**2 +\
        6.*(strain[:,3,...]**2 + strain[:,4,...]**2 + strain[:,5,...]**2)
    )

def equivalent_strain(strain: np.ndarray) -> np.ndarray:
    if strain.shape[1] == 6:
        return _equivalent_strain_from_full(strain)
    else:
        raise NotImplementedError("strain must be specified using full tensor")

def principal_values(tens: np.ndarray) -> np.ndarray:
    """
    Compute the principal stresses of a stress tensor - this is just the eigenvalue
    probelem at every integration point, so reshape the tensor appropriately,
    do the eigenvalue computation (which is symmetric) and return that result
    """
    mat = np.empty([tens.shape[0],3,3,tens.shape[-1]])

    #lazy mapping flattened symmetric tensor indices to
    #3x3 tensor matrix - not sure if eigvalsh requires
    #the upper half of the matrix, but this is only assigning
    #the view of the matrix, not copying so this is a pretty cheap
    #operation 
    mapping = {(0,0): 0, 
                (1,1): 1,
                (2,2): 2,
                (0,1): 3,
                (1,0): 3,
                (0,2): 5,
                (2,0): 5,
                (1,2): 4,
                (2,1): 4}

    for coord,idx in mapping.items():
        mat[:,coord[0],coord[1],:] = tens[:,idx,:]
    
    #we can use eigvalsh because the tensor is symmetric
    #have to do some swap axis here because numpy wants to compute the
    #values along the last axis, so swap, do eignvalsh, then swap back
    return np.linalg.eigvalsh(mat.swapaxes(1,-1)).swapaxes(-2,-1).squeeze()

def membrane_tensor(stress:np.ndarray,
                    thick: np.ndarray,
                    xs: np.ndarray) -> np.ndarray:
    """
    compute the membrane tensor at every point across
    the thickness, thus the integration is cumulative
    """
    return trapezoid(stress,x = xs[:,None,:],axis = 2)/thick[:,None]

        
def bending_tensor(stress: np.ndarray,
                   thick: np.ndarray,
                   xt: np.ndarray) -> np.ndarray:
    """ 
    compute the bending tensor, evaluates the integral definition according
    to ITER guidelines
    """

    xs = thick[:,np.newaxis]/2.0 - xt
    coeff = 6/(np.power(thick[:,None],2))
    bst = simpson(stress*xs[:,np.newaxis,:],x = xt[:,np.newaxis,:],axis = 2)
    return bst*coeff

def triaxility_factor(stress: np.ndarray) -> np.ndarray:

    ps = principal_values(stress)
    hydrostatic = 1./3.*ps.sum(axis = 1)
    octaheadral = _von_mises_from_full(stress).squeeze()
    return hydrostatic/octaheadral

def significant_strain(strain: np.ndarray) -> np.ndarray:
    return principal_values(strain).max(axis = 1)

class APDLIntegrate:
    """
    Class for integrating "stress" - really any finite dimensional field
    across a thickness. 

    Parameters
    -----------
    stress: np.ndarray
        (N*m)xD numpy array containing the stress at the cartesian "locations. D is the number
        of dimensions in the field, for example they could be the components of the stress
        tensor, which is symmetric and can be represented as a length 6 vector, or they could
        be one-dimensional, for scalar valued fields such as temperature. m is the number of integration
        points, and N is the number of points in the mesh we are integrating over
    locations: np.ndarray
        (N*m)xn numpy array containing the "locations of the stress field. n is the number
        of coordiante dimensions i.e. 1,2,3
    npoints: int
        the number of integration points through the thickness to itegrate across, "m".
    """

    def __init__(self,stress: np.ndarray,
                      locations: np.ndarray,
                      npoints: int):
        
        if stress.shape[0] != locations.shape[0]:
            raise ValueError('cannot integrate on uneven arrays')
        
        self.npoints = npoints
        self.stress = stress.reshape([-1,npoints,stress.shape[1]])
        self.stress = self.stress.swapaxes(-1,-2)

        self.locations = locations.reshape([-1,npoints,locations.shape[1]])
        self.locations = self.locations.swapaxes(-1,-2)

    @property
    def xs(self):
        return np.linalg.norm(self.locations - self.x1[...,None],axis = 1)

    @property
    def x1(self):
        return self.locations[:,:,0]
    
    @property
    def x2(self):
        return self.locations[:,:,-1]
    
    @property
    def thick(self):
        return np.linalg.norm(self.x2 - self.x1,axis = 1)
    
    def _membrane_average(self,output: np.ndarray,
                              averaged = True) -> np.ndarray:
        return _repeated_quantity(output,averaged,self.stress.shape[-1])
    
    def _bending_average(self,output: np.ndarray,
                             averaged = True) -> np.ndarray:
        return _linear_quantity(output,averaged,self.stress.shape[-1])
    
    def thickness_average(self) -> np.ndarray:
        """
        average value across the thickness
        """
        return self.membrane_tensor(averaged = True)

    def bending_tensor(self,averaged = True) -> np.ndarray:

        a = bending_tensor(self.stress,self.thick,
                            self.xs)
        return self._bending_average(a,averaged)

    def membrane_tensor(self,averaged = True) -> np.ndarray:

        a = membrane_tensor(self.stress,self.thick,
                            self.xs)
        return self._membrane_average(a,averaged)
    
    def linearized_principal_values(self,averaged = True) -> np.ndarray:
        """ 
        Compute the linearized principal stress, essentially just computes the lineraized
        stresses, and then computes the pinrciplate stresses of the linearized stresses.
        """

        pv = principal_values(self.stress)
        lin_pv = membrane_tensor(pv,self.thick,self.xs) + bending_tensor(pv,self.thick,self.xs)
        return self._bending_average(lin_pv,averaged)
    
    def membrane_stress_vm(self,averaged = True) -> np.ndarray:
        """
        convinience function for computing the von-mises membrane
        stresses
        """
        mt = self.membrane_tensor(averaged = True)
        vm = equivalent_stress(mt)
        return self._membrane_average(vm,averaged)
    
    def membrane_strain_equiv(self,averaged = True) -> np.ndarray:

        mt = self.membrane_tensor(averaged = True)
        equiv = equivalent_strain(mt)
        return self._membrane_average(equiv,averaged)
    
    def significant_mean_strain(self,averaged = True) -> np.ndarray:

        pv = principal_values(self.stress)
        membrane_pv = membrane_tensor(pv,self.thick,self.xs).max(axis = 1)
        return self._membrane_average(membrane_pv,averaged)
    
    def significant_linear_strain(self,averaged = True) -> np.ndarray:

        lpv = self.linearized_principal_values(averaged = True)
        a = lpv.max(axis = 1)
        return self._bending_average(a,averaged)
    
    def significant_total_strain(self) -> np.ndarray:
        return significant_strain(self.stress)

    def bending_stress_vm(self,averaged = True) -> np.ndarray:
        """
        convinience function for computing the von-mises bending
        stresses
        """
        bt = self.bending_tensor(averaged = True)
        vm = equivalent_stress(bt)
        return self._bending_average(vm,averaged)

    def linearized_triaxiality_factor(self,averaged = True) -> np.ndarray:
        """
        function for computing the "triaxility factor"
        """
        tf = triaxility_factor(self.stress)
        lin = membrane_tensor(tf[:,np.newaxis,:],self.thick,self.xs) + \
              bending_tensor(tf[:,np.newaxis,:],self.thick,self.xs)
        return self._membrane_average(lin.squeeze(),averaged)

        


