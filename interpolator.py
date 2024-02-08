import numpy as np 
from scipy.interpolate import LinearNDInterpolator, griddata
import scipy.io as sio


def interp_data_RZ_to_tri(R,Z,data,verts,cells,method='linear'):
    """
    Interpolate data defined in R,Z space to tri grid.
   
    Parameters
    -----------
    R : array float
        Radial locations.
    Z : array float
        Vertical locations
    data : matrix float
        Values on R, Z points to interpolate to grid N(data)xN(time) or N(data)
    verts : matrix float
        Vertices of target grid N(verts)x2.
    cells : matrx float
        Cells in terms of vertex numbers N(cells)x3.
    method : str
        Method to use (cubic, linear)
   
    Returns
    --------
    data_new : array float
        Values on cells interpolated to new grid N(cells)xN(time).
    """
    if len(data.shape) == 1:
        data = np.reshape(data,(len(data),1))
    Nt = data.shape[1]
    data_new = np.zeros((cells.shape[0],Nt))
    # cell centers
    # R_cells = np.mean(verts[cells,0],axis=1)
    # Z_cells = np.mean(verts[cells,1],axis=1)

    # R_cells = np.mean(verts[cells,0],axis=1)

    # print(verts)

    # print(cells)

    # R_cells = np.zeros((7923,1))
    # Z_cells = np.zeros((7923,1))

    cells = cells - 1

    R_cells = np.mean(verts[cells,0],axis=1)
    Z_cells = np.mean(verts[cells,1],axis=1)

    # print('jaime')
    # print(R_cells.shape)

    # for i in range(7923):
    #     R_cells[i, 0] = np.mean(np.array([verts[cells[i, 0], 0], verts[cells[i, 1], 0], verts[cells[i, 2], 0]]))
    #     Z_cells[i, 0] = np.mean(np.array([verts[cells[i, 0], 1], verts[cells[i, 1], 1], verts[cells[i, 2], 1]]))

    #print(verts[])

    # R_cells = np.mean(np.array([verts[cells[:,0], 0],verts[cells[:,1], 0],verts[cells[:,2], 0]]), axis=0)
    # print(R_cells.shape)
    # Z_cells = np.mean(np.array([cells[verts[:,0],1],cells[verts[:,1],1],cells[verts[:,2],1]]),axis=0)
    # interpolation
    for tidx in np.arange(Nt):
        if method == 'linear':
            interp_func = LinearNDInterpolator(list(zip(R,Z)),np.ravel(data[:,tidx]))
            data_new[:,tidx] = np.nan_to_num(interp_func(list(zip(R_cells,Z_cells))),nan=0.0)
        elif method == 'cubic':
            data_new[:,tidx] = griddata(list(zip(R,Z)),np.ravel(data[:,tidx], order='F'),\
                list(zip(R_cells, Z_cells)), method='cubic', fill_value=0.0)
        else:
            raise Exception('mehtod not recognized.')
    return data_new

# centroid of sqaure from b2
data_sq = sio.loadmat('SOLPS-ITER/geom_sq.mat')
# making long array from corners of square grid
# (meaning, going to 98x38x4 to 98*38x4)
long_arr_crx = np.reshape(data_sq['geom_3']['crx'][0][0], (98*38, 4))
long_arr_cry = np.reshape(data_sq['geom_3']['cry'][0][0], (98*38, 4))
long_arr_centroid = np.zeros((98*38, 2))
# simply calculating centroid
for i in range(98*38):
        long_arr_centroid[i, 0] = np.mean(long_arr_crx[i, :])
        long_arr_centroid[i, 1] = np.mean(long_arr_cry[i, :])

# taking data from b2
data_te = sio.loadmat('SOLPS-ITER/te_simul.mat')
# making long array from corners of b2 grid
Te_long_arr = np.reshape(data_te['Te_simul'], ((98*38, 1))) #np.reshape(data_b2['state_3']['te'][0][0], (98*38, 1))
# print(Te_long_arr.shape)
data_ne = sio.loadmat('SOLPS-ITER/ne_simul.mat')
ne_long_arr = np.reshape(data_ne['ne_simul'], ((98*38, 1))) #np.reshape(data_b2['state_3']['te'][0][0], (98*38, 1))

data_he = sio.loadmat('SOLPS-ITER/heplus_simul.mat')
He_long_arr = np.reshape(data_he['Heplus_simul'], ((98*38, 1))) #np.reshape(data_b2['state_3']['te'][0][0], (98*38, 1))

# load the vertices and cells of the target grid
data_tri = sio.loadmat('SOLPS-ITER/geom_tri.mat')
verts = data_tri['geomtri_3']['nodes'][0][0]
# print(verts.shape)
cells = data_tri['geomtri_3']['cells'][0][0]
# print(cells.shape)

Te_tri_arr = interp_data_RZ_to_tri(long_arr_centroid[:,0], long_arr_centroid[:,1], Te_long_arr, verts, cells)

print(Te_tri_arr.shape)

ne_tri_arr = interp_data_RZ_to_tri(long_arr_centroid[:,0], long_arr_centroid[:,1], ne_long_arr, verts, cells)

print(ne_tri_arr.shape)

Heplus_tri_arr = interp_data_RZ_to_tri(long_arr_centroid[:,0], long_arr_centroid[:,1], He_long_arr, verts, cells)

print(Heplus_tri_arr.shape)

# save the data in .mat format

sio.savemat('SOLPS-ITER/te_tri.mat', {'te_tri': Te_tri_arr})
sio.savemat('SOLPS-ITER/ne_tri.mat', {'ne_tri': ne_tri_arr})
sio.savemat('SOLPS-ITER/heplus_tri.mat', {'heplus_tri': Heplus_tri_arr})

