#some functions used 
#make sure you have excuted this module before you import it
import netCDF4 
from wrf import to_np,getvar
import numpy as np
def nc_read1(filename,var):
    with netCDF4.Dataset(filename,'r') as ncfile:
        data = ncfile.variables[var][:].squeeze()
        return data

def getTClocation(file):
    """
    Get the location of the TC center from the slp field
    :param file: path to the netCDF file
    :return: tuple of (iTC, jTC) indices of the TC center
    """
    # Use netCDF4 to read the file
    data = netCDF4.Dataset(file)
    
    # Get the sea level pressure (slp) variable
    slp = to_np(getvar(data, 'slp', timeidx=0))
    
    # Find the minimum slp value and its indices
    slpMin = np.min(slp[:, :])
    indexMin = np.argwhere(slp[:, :] == slpMin)
    
    # Extract the i and j indices of the TC center
    jTC = indexMin[0][0]
    iTC = indexMin[0][1]
    
    return iTC, jTC