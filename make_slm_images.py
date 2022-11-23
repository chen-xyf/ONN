import numpy as np
import cupy as cp
from scipy import ndimage
import cProfile
import pstats

##########
# SANTEC #
##########

slm_resX = 1920
slm_resY = 1080

slm_resX_actual = 1440
slm_resY_actual = 1050

slm_xc = slm_resX_actual//2
slm_yc = slm_resY_actual//2

slm_w = 1406
slm_h = 775

slm_x0 = slm_xc-(slm_w//2)
slm_y0 = slm_yc-(slm_h//2)

slm_sig_xc = 222
slm_err_xc = int(5*slm_resX_actual//6)

slm_R1_h = 30
slm_gap = 85

slm_R2_w = 55

slm_sig_w = 396
slm_sig_h = 390

slm_err_w = 350
slm_err_h = slm_sig_h

slm_err_w_rot = 350
slm_err_h_rot = 390

slm_edge = (slm_h - (slm_sig_h+slm_gap+slm_R1_h))//2
slm_edge_err = (slm_h - (slm_err_h_rot+slm_gap+slm_R1_h))//2

slm_s = np.s_[slm_x0:slm_x0+slm_w, slm_y0:slm_y0+slm_h]

slm_R1_s = np.s_[slm_sig_xc-slm_sig_w//2:slm_sig_xc+slm_sig_w//2,
                 420+slm_sig_h:420+slm_sig_h+slm_R1_h]
slm_sig_s = np.s_[slm_sig_xc-slm_sig_w//2:slm_sig_xc+slm_sig_w//2,
                  388:388+slm_sig_h]
slm_err_s = np.s_[slm_err_xc-slm_err_w_rot//2:slm_err_xc+slm_err_w_rot//2,
                  slm_y0+slm_edge_err+slm_R1_h+slm_gap:slm_y0+slm_h-slm_edge_err]

slm_R2_pos_s = np.s_[430:430+slm_R2_w, slm_y0+slm_edge+slm_R1_h+slm_gap:slm_y0+slm_h-slm_edge]
slm_R2_neg_s = np.s_[:slm_R2_w, slm_y0+slm_edge+slm_R1_h+slm_gap:slm_y0+slm_h-slm_edge]


#######
# DMD #
#######

dmd_resX = 1920
dmd_resY = 1080

dmd_w = 1920
dmd_h = 1080

dmd_xc = dmd_resX//2
dmd_yc = dmd_resY//2

dmd_x0 = dmd_xc-(dmd_w//2)
dmd_y0 = dmd_yc-(dmd_h//2)

dmd_sig_xc = dmd_w//6 - 20
dmd_err_xc = int(5*dmd_w//6) + 28

dmd_err_yc = 608

dmd_sig_w = 587
dmd_sig_h = int(slm_sig_h * dmd_h/slm_h)-1

dmd_err_w = int(slm_err_w * dmd_w/slm_w)

dmd_R1_h = int(slm_R1_h * dmd_h/slm_h)
dmd_R2_w = 50
dmd_gap = int(slm_gap * dmd_h/slm_h)
dmd_edge = (dmd_h - (dmd_sig_h+dmd_gap+dmd_R1_h))//2

dmd_s = np.s_[dmd_x0:dmd_x0+dmd_w, dmd_y0:dmd_y0+dmd_h]

dmd_R1_s = np.s_[920:920+dmd_R1_h, dmd_sig_xc-dmd_sig_w//2:dmd_sig_xc+dmd_sig_w//2+1, :]
dmd_sig_s = np.s_[dmd_sig_xc-dmd_sig_w//2:dmd_sig_xc+dmd_sig_w//2+1,
                  dmd_y0+dmd_edge+1+dmd_R1_h+dmd_gap:dmd_y0+dmd_h-dmd_edge]
dmd_err_s = np.s_[dmd_err_xc-dmd_err_w//2:dmd_err_xc+dmd_err_w//2+1,
                  dmd_err_yc-dmd_sig_h//2:dmd_err_yc+dmd_sig_h//2]

# used in code to create batch of dmd images quickly
dmd_sig_s_multi = np.s_[:, dmd_sig_xc-dmd_sig_w//2:dmd_sig_xc+dmd_sig_w//2+1,
                        dmd_y0+dmd_edge+1+dmd_R1_h+dmd_gap:dmd_y0+dmd_h-dmd_edge, :]
dmd_err_s_multi = np.s_[:, dmd_err_xc-dmd_err_w//2:dmd_err_xc+dmd_err_w//2+1,
                        dmd_err_yc-dmd_sig_h//2:dmd_err_yc+dmd_sig_h//2, :]

dmd_R2_pos_s = np.s_[550:550+dmd_R2_w, dmd_y0+dmd_edge+1+dmd_R1_h+dmd_gap:dmd_y0+dmd_h-dmd_edge]
dmd_R2_neg_s = np.s_[:dmd_R2_w, dmd_y0+dmd_edge+1+dmd_R1_h+dmd_gap:dmd_y0+dmd_h-dmd_edge]


##############
# MEADOWLARK #
##############

slm2_resX = 1920
slm2_resY = 1152

slm2_w = 1920
slm2_h = 1152

slm2_xc = 1104
slm2_yc = 322

slm2_sig_w = 250
slm2_sig_h = 440

slm2_R1_h = 1
slm2_R1_y0 = 0

slm2_sig_s = np.s_[slm2_xc-slm2_sig_w//2:slm2_xc+slm2_sig_w//2, slm2_yc-slm2_sig_h//2:slm2_yc+slm2_sig_h//2]
slm2_R1_s = np.s_[slm2_xc-slm2_sig_w//2:slm2_xc+slm2_sig_w//2, slm2_R1_y0:slm2_R1_y0+slm2_R1_h]



######################################

# create index arrays
x_SLM = slm_resX_actual//2
y_SLM = slm_resY_actual//2
X_array = np.arange(-x_SLM, x_SLM)
Y_array = np.arange(-y_SLM, y_SLM)
Y, X = np.meshgrid(Y_array, X_array)

# load the phase to GL LUT
GL_lut_arr = np.load('./tools/Santec_LUT_NEW_4PI.npy')
# determine how many GLs we have
max_GL = GL_lut_arr.shape[0]
# calculate the step in phase between each GL
d_phase = GL_lut_arr[-1, 0]/max_GL

gpu_LUT_arr = cp.asarray(GL_lut_arr, dtype='float32')

gr = 2*np.pi*(-X+Y)/(10+1e-5)
gpu_gr = cp.asarray(gr, dtype='float32')

slm_wf = np.load("./tools/slm_wf.npy")
gpu_slm_wf = cp.array(slm_wf)

slm2_wf = np.load("./tools/slm2_wf.npy").T
gpu_slm2_wf = cp.array(slm2_wf)

def gpu_arcsinc(y):
    z = 1.*(1-y)
    return 1 - ((cp.sqrt((12*z)/(1-(0.20409*z)+cp.sqrt(1-(0.792*z)-(0.0318*z*z)))))/cp.pi)


# create index arrays
x_SLM2 = slm2_w//2
y_SLM2 = slm2_h//2
X_array2 = np.arange(-x_SLM2, x_SLM2)
Y_array2 = np.arange(-y_SLM2, y_SLM2)
Y2, X2 = np.meshgrid(Y_array2, X_array2)

gr2 = 2*np.pi*(-X2+Y2)/(10+1e-5)
gpu_gr2 = cp.asarray(gr2, dtype='float32')


######################################

phi_sig_corr = np.load("./tools/phi_corr_w1_nov03_smallbeam.npy")
gpu_phi_sig_corr = cp.array(phi_sig_corr)

phi_err_corr = np.load("./tools/phi_err_w1_aug22.npy")
phi_err_corr = np.flip(phi_err_corr, axis=0)
phi_err_corr = np.flip(phi_err_corr, axis=1)
gpu_phi_err_corr = cp.array(phi_err_corr)

uppers_err_2km = np.load("./tools/uppers_err_2km_w1_aug22.npy")

phi_sig_corr_2 = np.load("./tools/phi_corr_2.npy")
phi_sig_corr_2 = np.flip(phi_sig_corr_2, axis=0)
phi_sig_corr_2 = np.flip(phi_sig_corr_2, axis=1)
gpu_phi_sig_corr_2 = cp.array(phi_sig_corr_2)

# uppers_w2_mk = np.load("./tools/uppers_w2_mk_aug22.npy")

# slm_R2_pos_phase_block = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/"
#                                  "ONN/tools/slm_R2_pos_phase_block.npy")
# gpu_slm_R2_pos_phase_block = cp.array(slm_R2_pos_phase_block)
#
# slm_R2_neg_phase_block = np.load("C:/Users/spall/OneDrive - Nexus365/Code/JS/PycharmProjects/"
#                                  "ONN/tools/slm_R2_neg_phase_block.npy")
# gpu_slm_R2_neg_phase_block = cp.array(slm_R2_neg_phase_block)


slm_gap_x = 5
slm_gap_y = 2
slm2_gap_x = 10
slm2_gap_y = 4
dmd_gaps = True

# SLM1 error
tilt = 22
tilt_levels = cp.array([-(i-slm_resY_actual//2)//tilt for i in range(slm_resY_actual)])
tilt_row_indices, tilt_column_indices = cp.ogrid[:slm_resY_actual, :slm_resX_actual//2]
tilt_levels[tilt_levels < 0] += slm_resX_actual//2
tilt_column_indices = tilt_column_indices - tilt_levels[:, cp.newaxis]

tilt = -65
tilt_levels = cp.array([i//tilt for i in range(slm_resX_actual//2)])
tilt_vert_row_indices, tilt_vert_column_indices = cp.ogrid[slm_resX_actual // 2:slm_resX_actual, :slm_resY_actual]
tilt_levels[tilt_levels < 0] += slm_resY_actual
tilt_vert_column_indices = tilt_vert_column_indices.copy() - tilt_levels[:, cp.newaxis]

# SLM2
tilt2 = -24
tilt2_levels = cp.array([i//tilt2 for i in range(slm2_resY)])
tilt2_levels[tilt2_levels < 0] += slm2_resX
tilt2_row_indices, tilt2_column_indices = cp.ogrid[:slm2_resY, :slm2_resX]
tilt2_column_indices = tilt2_column_indices - tilt2_levels[:, cp.newaxis]

tilt2 = -62
tilt2_levels = cp.array([i//tilt2 for i in range(slm2_resX)])
tilt2_levels[tilt2_levels < 0] += slm2_resY
tilt2_vert_row_indices, tilt2_vert_column_indices = cp.ogrid[:slm2_resX, :slm2_resY]
tilt2_vert_column_indices = tilt2_vert_column_indices - tilt2_levels[:, cp.newaxis]

# DMD
tilt = -30
tilt_levels_dmd = cp.array([ii//tilt for ii in range(dmd_h)])
rows_indices_dmd, columns_indices_dmd = cp.ogrid[:dmd_h, :dmd_w//2]
columns_indices_dmd = columns_indices_dmd - tilt_levels_dmd[:, cp.newaxis]

tilt = -65
tilt_levels_dmd = cp.array([ii//tilt for ii in range(dmd_w//2)])
rows_vert_indices_dmd, columns_vert_indices_dmd = cp.ogrid[dmd_w//2:dmd_w, :dmd_h]
columns_vert_indices_dmd = columns_vert_indices_dmd - tilt_levels_dmd[:, cp.newaxis]

######################################

global n, m, k, num_frames
global cols_to_keep, rows_to_keep, slm_block_w, slm_block_h
global cols_to_keep2, rows_to_keep2, slm2_block_w, slm2_block_h
global dmd_block_w, dmd_block_h, gpu_dmd_centers_x, gpu_dmd_centers_y
global ws, dmd_xc_shifted, y_blocks_multi_cp
global slm_err_block_w, slm_err_block_h, err_cols_to_del, err_rows_to_del
global dmd_err_block_w, dmd_err_block_h, gpu_dmd_err_centers_x, gpu_dmd_err_centers_y
global ws_err, dmd_xc_shifted_err, y_blocks_multi_cp_err
global gpu_err_a_aoi, gpu_err_phi_aoi


def update_params(_n, _m, _k, _num_frames):

    global n, m, k, num_frames
    global cols_to_keep, rows_to_keep, slm_block_w, slm_block_h
    global cols_to_keep2, rows_to_keep2, slm2_block_w, slm2_block_h
    global dmd_block_w, dmd_block_h, gpu_dmd_centers_x, gpu_dmd_centers_y
    global ws, dmd_xc_shifted, y_blocks_multi_cp
    global slm_err_block_w, slm_err_block_h, err_cols_to_del, err_rows_to_del
    global dmd_err_block_w, dmd_err_block_h, gpu_dmd_err_centers_x, gpu_dmd_err_centers_y
    global ws_err, dmd_xc_shifted_err, y_blocks_multi_cp_err
    global gpu_err_a_aoi, gpu_err_phi_aoi

    n = _n
    m = _m
    k = _k
    num_frames = _num_frames

    slm_block_w = int(slm_sig_w / n) + 1
    cols_to_del = np.linspace(0, slm_block_w*n - 1, (slm_block_w * n) - slm_sig_w).astype(np.int32)
    cols_to_keep = np.delete(np.arange(slm_block_w*n), cols_to_del)

    slm_block_h = int(slm_sig_h / m) + 1
    rows_to_del = np.linspace(0, slm_block_h*m - 1, (slm_block_h * m) - slm_sig_h).astype(np.int32)
    rows_to_keep = np.delete(np.arange(slm_block_h*m), rows_to_del)

    slm2_block_w = int(slm2_sig_w / k) + 1
    cols_to_del2 = np.linspace(0, slm2_block_w*k - 1, (slm2_block_w * k) - slm2_sig_w).astype(np.int32)
    cols_to_keep2 = np.delete(np.arange(slm2_block_w*k), cols_to_del2)

    slm2_block_h = int(slm2_sig_h / m) + 1
    rows_to_del2 = np.linspace(0, slm2_block_h*m - 1, (slm2_block_h * m) - slm2_sig_h).astype(np.int32)
    rows_to_keep2 = np.delete(np.arange(slm2_block_h*m), rows_to_del2)

    slm_edges_x = np.linspace(0, slm_sig_w, n+1)
    slm_centers_x = np.array([(slm_edges_x[ii]+slm_edges_x[ii+1])/2 for ii in range(n)]).astype(int)

    slm_edges_y = np.linspace(0, slm_sig_h, m+1)
    slm_centers_y = np.array([(slm_edges_y[ii]+slm_edges_y[ii+1])/2 for ii in range(m)]).astype(int)

    if dmd_gaps:
        dmd_block_w = int((slm_block_w-2) * .9 * dmd_w / slm_w)
        dmd_block_h = int((slm_block_h-1) * .45 * dmd_h / slm_h)
    else:
        dmd_block_w = int(slm_block_w * dmd_w / slm_w)
        dmd_block_h = int(slm_block_h * dmd_h / slm_h)

    if dmd_block_h % 2 == 0:
        dmd_block_h -= 1

    dmd_centers_x = (slm_centers_x * dmd_w / slm_w).astype(int)
    dmd_centers_y = (slm_centers_y * dmd_h / slm_h).astype(int)
    dmd_centers_x_grid, dmd_centers_y_grid = np.meshgrid(dmd_centers_x, dmd_centers_y)

    gpu_dmd_centers_x = cp.array(dmd_centers_x_grid)
    gpu_dmd_centers_y = cp.array(dmd_centers_y_grid)

    r = cp.array([int((-1) ** ii * cp.ceil(ii / 2)) for ii in range(dmd_block_w+1)])
    ws, ns = cp.meshgrid(cp.arange(dmd_block_w+1), cp.arange(n), indexing='ij')
    dmd_xc_shifted = gpu_dmd_centers_x[0, :] - r[:, cp.newaxis]

    y_blocks = cp.zeros(dmd_sig_h, dtype='bool')
    for j in range(dmd_block_h // 2 + 1):
        y_blocks[gpu_dmd_centers_y[:, 0]-j] = 1
        y_blocks[gpu_dmd_centers_y[:, 0]+j] = 1

    y_blocks_multi_cp = y_blocks[cp.newaxis, :].astype(cp.bool_)

    # params for error block

    slm_err_block_w = int(slm_err_w / (2*k)) + 1
    err_cols_to_del = np.linspace(0, slm_err_block_w*(2*k) - 1, (slm_err_block_w * (2*k)) - slm_err_w).astype(np.int32)

    slm_err_block_h = int(slm_err_h / m) + 1
    err_rows_to_del = np.linspace(0, slm_err_block_h*m - 1, (slm_err_block_h * m) - slm_err_h).astype(np.int32)

    slm_err_edges_x = np.linspace(0, slm_err_w, (2*k)+1)
    slm_err_centers_x = np.array([(slm_err_edges_x[ii]+slm_err_edges_x[ii+1])/2 for ii in range(2*k)]).astype(int)

    slm_err_edges_y = np.linspace(0, slm_err_h, m+1)
    slm_err_centers_y = np.array([(slm_err_edges_y[jj]+slm_err_edges_y[jj+1])/2 for jj in range(m)]).astype(int)

    if dmd_gaps:
        dmd_err_block_w = int((slm_err_block_w-2) * .9 * dmd_w / slm_w)
        dmd_err_block_h = int((slm_err_block_h-1) * .45 * dmd_h / slm_h)
    else:
        dmd_err_block_w = int((slm_err_block_w-2) * dmd_w / slm_w)
        dmd_err_block_h = int((slm_err_block_h-2) * dmd_w / slm_w)

    dmd_err_centers_x = (slm_err_centers_x * dmd_w / slm_w).astype(int)
    dmd_err_centers_y = (slm_err_centers_y * dmd_h / slm_h).astype(int)
    dmd_err_centers_x_grid, dmd_err_centers_y_grid = np.meshgrid(dmd_err_centers_x, dmd_err_centers_y)

    gpu_dmd_err_centers_x = cp.array(dmd_err_centers_x_grid)
    gpu_dmd_err_centers_y = cp.array(dmd_err_centers_y_grid)

    r_err = cp.array([int((-1) ** ii * cp.ceil(ii / 2)) for ii in range(dmd_err_block_w+1)])
    ws_err, ns_err = cp.meshgrid(cp.arange(dmd_err_block_w+1), cp.arange(2*k), indexing='ij')
    dmd_xc_shifted_err = gpu_dmd_err_centers_x[0, :] - r_err[:, cp.newaxis]

    y_blocks = cp.zeros(dmd_sig_h, dtype='bool')
    for j in range(dmd_err_block_h // 2 + 1):
        y_blocks[gpu_dmd_err_centers_y[:, 0]-j] = 1
        y_blocks[gpu_dmd_err_centers_y[:, 0]+j] = 1

    y_blocks_multi_cp_err = y_blocks[cp.newaxis, :].astype(cp.bool_)

    err_phi = np.zeros((2*k, m))*0.
    err_phi[::2, :] = np.pi
    err_phi = np.flip(err_phi, axis=0)
    err_phi = np.flip(err_phi, axis=1)
    err_phi_aoi = np.repeat(err_phi, slm_err_block_w, axis=0)
    err_phi_aoi = np.repeat(err_phi_aoi, slm_err_block_h, axis=1)
    err_phi_aoi = np.delete(err_phi_aoi, err_cols_to_del, 0)
    err_phi_aoi = np.delete(err_phi_aoi, err_rows_to_del, 1)

    err_a = np.ones((2*k, m))*1. # * uppers_err_2km.copy()
    # err_a[0, :] = 0.1
    # err_a[:, 1] = 0.1
    err_a = np.flip(err_a, axis=0)
    err_a = np.flip(err_a, axis=1)
    err_a_aoi = np.repeat(err_a, slm_err_block_w, axis=0)
    err_a_aoi = np.repeat(err_a_aoi, slm_block_h, axis=1)
    for i in range(2*k):
        err_a_aoi[i*slm_err_block_w:(i*slm_err_block_w)+slm_gap_x, :] = 0
        err_a_aoi[((i+1)*slm_err_block_w)-slm_gap_x:((i+1)*slm_err_block_w), :] = 0
    for j in range(m):
        err_a_aoi[:, j*slm_block_h:(j*slm_block_h)+slm_gap_y] = 0
        err_a_aoi[:, ((j+1)*slm_block_h)-slm_gap_y:((j+1)*slm_block_h)] = 0
    err_a_aoi = np.delete(err_a_aoi, err_cols_to_del, 0)
    err_a_aoi = np.delete(err_a_aoi, rows_to_del, 1)

    gpu_err_a_aoi = cp.array(err_a_aoi)
    gpu_err_phi_aoi = cp.array(err_phi_aoi)

    return dmd_block_w, dmd_err_block_w


def make_slm1_rgb(target_a, target_phi):

    # profile = cProfile.Profile()
    # profile.enable()

    gpu_a = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')
    gpu_phi = cp.zeros((slm_resX_actual, slm_resY_actual), dtype='float32')

    a_aoi = cp.repeat(target_a, slm_block_w, axis=0)
    a_aoi = cp.repeat(a_aoi, slm_block_h, axis=1)
    for ii in range(n):
        a_aoi[ii*slm_block_w:(ii*slm_block_w)+slm_gap_x, :] = 0
        a_aoi[((ii+1)*slm_block_w)-slm_gap_x:((ii+1)*slm_block_w), :] = 0
    for jj in range(m):
        a_aoi[:, jj*slm_block_h:(jj*slm_block_h)+slm_gap_y] = 0
        a_aoi[:, ((jj+1)*slm_block_h)-slm_gap_y:((jj+1)*slm_block_h)] = 0
    a_aoi = a_aoi[cols_to_keep, :][:, rows_to_keep]

    phi_aoi = cp.repeat(target_phi, slm_block_w, axis=0)
    phi_aoi = cp.repeat(phi_aoi, slm_block_h, axis=1)
    phi_aoi = phi_aoi[cols_to_keep, :][:, rows_to_keep]

    gpu_a[slm_sig_s] = a_aoi
    gpu_phi[slm_sig_s] = phi_aoi + gpu_phi_sig_corr

    gpu_a[slm_R1_s] = 1.

    gpu_a[slm_err_s] = gpu_err_a_aoi
    gpu_phi[slm_err_s] = gpu_err_phi_aoi #+ gpu_phi_err_corr

    # gpu_a[slm_R2_pos_s] = 0.55  # to balance pos and neg
    # gpu_phi[slm_R2_pos_s] = gpu_slm_R2_pos_phase_block
    # gpu_a[slm_R2_neg_s] = 1.
    # gpu_phi[slm_R2_neg_s] = gpu_slm_R2_neg_phase_block + np.pi

    gpu_phi[slm_s] += gpu_slm_wf

    gpu_a = cp.maximum(gpu_a, 1e-16, dtype='float32')
    gpu_a = cp.minimum(gpu_a, 1 - 1e-4, dtype='float32')

    gpu_a[gpu_a.shape[0]//2:, :] = gpu_a[gpu_a.shape[0]//2:, :][tilt_column_indices, tilt_row_indices].T
    gpu_phi[gpu_a.shape[0]//2:, :] = gpu_phi[gpu_a.shape[0]//2:, :][tilt_column_indices, tilt_row_indices].T
    gpu_a[slm_resX_actual//2:, :] = gpu_a[slm_resX_actual//2:, :][tilt_vert_row_indices, tilt_vert_column_indices]
    gpu_phi[slm_resX_actual//2:, :] = gpu_phi[slm_resX_actual//2:, :][tilt_vert_row_indices, tilt_vert_column_indices]

    gpu_m = gpu_arcsinc(gpu_a)
    gpu_f = gpu_phi + cp.pi*(1-gpu_m)
    gpu_sig = gpu_m*cp.mod(gpu_f+gpu_gr, 2*cp.pi)

    # phase-GL LUT
    gpu_idx_phase = (gpu_sig / d_phase).astype(int)
    gpu_idx_phase = cp.clip(gpu_idx_phase, 0, max_GL-1).astype(int)
    gpu_o_gl = gpu_LUT_arr[gpu_idx_phase, 1]

    gpu_out = cp.zeros((slm_resX, slm_resY, 1), dtype='float16')
    gpu_out[:slm_resX_actual, :slm_resY_actual, 0] = gpu_o_gl
    gpu_out = gpu_out.transpose(1, 0, 2)

    # 10-bit santec encoding
    gpu_r_array = gpu_out // 128
    gpu_g_array = gpu_out // 16 - gpu_r_array * 8
    gpu_b_array = gpu_out - (gpu_out // 16) * 16

    gpu_r_array = gpu_r_array * 32
    gpu_g_array = gpu_g_array * 32
    gpu_b_array = gpu_b_array * 16

    gpu_r_array = gpu_r_array.astype(cp.uint8)
    gpu_g_array = gpu_g_array.astype(cp.uint8)
    gpu_b_array = gpu_b_array.astype(cp.uint8)

    gpu_color_array = cp.concatenate((gpu_r_array, gpu_g_array, gpu_b_array), axis=2)
    gpu_color_array.astype(cp.uint8)

    # del gpu_r_array, gpu_g_array, gpu_b_array, gpu_out
    # del gpu_idx_phase, gpu_sig, gpu_m, gpu_a, gpu_f, gpu_phi

    # noinspection PyProtectedMember
    # cp._default_memory_pool.free_all_blocks()

    # profile.disable()
    # ps = pstats.Stats(profile)
    # ps.print_stats()

    return gpu_color_array


def make_slm2_rgb(target_a, target_phi):

    target_a = cp.flip(target_a, axis=0)
    target_a = cp.flip(target_a, axis=1)
    target_phi = cp.flip(target_phi, axis=0)
    target_phi = cp.flip(target_phi, axis=1)

    a_aoi = cp.repeat(target_a, slm2_block_w, axis=0)
    a_aoi = cp.repeat(a_aoi, slm2_block_h, axis=1)
    for ii in range(k):
        a_aoi[ii*slm2_block_w:(ii*slm2_block_w)+slm2_gap_x, :] = 0
        a_aoi[((ii+1)*slm2_block_w)-slm2_gap_x:((ii+1)*slm2_block_w), :] = 0
    for jj in range(m):
        a_aoi[:, jj*slm2_block_h:(jj*slm2_block_h)+slm2_gap_y] = 0
        a_aoi[:, ((jj+1)*slm2_block_h)-slm2_gap_y:((jj+1)*slm2_block_h)] = 0
    a_aoi = a_aoi[cols_to_keep2, :][:, rows_to_keep2]

    phi_aoi = cp.repeat(target_phi, slm2_block_w, axis=0)
    phi_aoi = cp.repeat(phi_aoi, slm2_block_h, axis=1)
    phi_aoi = phi_aoi[cols_to_keep2, :][:, rows_to_keep2]

    gpu_a = cp.zeros((slm2_resX, slm2_resY), dtype='float32')
    gpu_phi = cp.zeros((slm2_resX, slm2_resY), dtype='float32')
    gpu_phi += gpu_slm2_wf.copy()

    gpu_a[slm2_sig_s] = a_aoi
    gpu_phi[slm2_sig_s] += phi_aoi + gpu_phi_sig_corr_2.copy()

    gpu_a[879:929, 40:70] = 1.

    gpu_a = gpu_a[tilt2_column_indices, tilt2_row_indices].T
    gpu_phi = gpu_phi[tilt2_column_indices, tilt2_row_indices].T
    gpu_a = gpu_a[tilt2_vert_row_indices, tilt2_vert_column_indices]
    gpu_phi = gpu_phi[tilt2_vert_row_indices, tilt2_vert_column_indices]

    gpu_a = cp.maximum(gpu_a, 1e-16, dtype='float32')
    gpu_a = cp.minimum(gpu_a, 1 - 1e-5, dtype='float32')

    gpu_m = gpu_arcsinc(gpu_a)
    gpu_f = gpu_phi + cp.pi * (1-gpu_m)
    gpu_o = gpu_m*cp.mod(gpu_f+gpu_gr2, 2*cp.pi)

    gpu_gl = (gpu_o * 255 / (2 * cp.pi)).astype(cp.uint8).T
    gpu_gl = cp.repeat(gpu_gl[..., None], 3, -1).astype(cp.uint8)

    # del gpu_m, gpu_a, gpu_f, gpu_phi, gpu_o

    # noinspection PyProtectedMember
    # cp._default_memory_pool.free_all_blocks()

    return gpu_gl


def make_dmd_batch(vecs, errs=None, marker=False):

    if marker:
        imgs = cp.zeros((1080, 1920, 3), dtype=cp.uint8)
        imgs[dmd_R1_s] = 255

        imgs = cp.flip(imgs, axis=0)
        imgs = cp.flip(imgs, axis=1)

        return imgs

    if errs is None:
        errs = cp.zeros((num_frames, 2*k))

    elif errs.shape[1] == k:
        errs = cp.repeat(errs, 2, axis=1)
        errs[:, ::2] *= (errs[:, ::2] < 0).astype(cp.int)
        errs[:, 1::2] *= (errs[:, 1::2] > 0).astype(cp.int)
        errs = cp.abs(errs)

    errs = cp.flip(errs, axis=1)

    assert vecs.shape == (num_frames, n)
    assert errs.shape == (num_frames, 2*k)

    def find_1d_pattern(vec):
        target = vec[cp.newaxis, :].astype(cp.uint8).repeat(dmd_block_w + 1, axis=0)
        mask = (target > cp.arange(dmd_block_w + 1)[:, cp.newaxis])
        mapped = cp.zeros((dmd_block_w + 1, dmd_sig_w), dtype='bool')
        mapped[ws, dmd_xc_shifted] = mask
        out = mapped.sum(axis=0).astype(cp.bool)
        return out

    def find_1d_pattern_err(vec):
        target = vec[cp.newaxis, :].astype(cp.uint8).repeat(dmd_err_block_w + 1, axis=0)
        mask = (target > cp.arange(dmd_err_block_w + 1)[:, cp.newaxis])
        mapped = cp.zeros((dmd_err_block_w + 1, dmd_err_w), dtype='bool')
        mapped[ws_err, dmd_xc_shifted_err] = mask
        out = mapped.sum(axis=0).astype(cp.bool)
        return out

    outxs = cp.empty((num_frames, dmd_sig_w), dtype='bool')
    for indx, vec in enumerate(vecs):
        outxs[indx, :] = find_1d_pattern(vec*dmd_block_w)
    sig_aoi = cp.einsum('eo,eu->eou', outxs, y_blocks_multi_cp).astype(cp.bool)
    sig_aoi = sig_aoi.reshape((num_frames, dmd_sig_w, dmd_sig_h)).astype(cp.uint8)[None, ...].repeat(3, axis=0)
    sig_aoi *= 255
    sig_aoi = cp.transpose(sig_aoi, (1, 2, 3, 0))

    outxs = cp.empty((num_frames, dmd_err_w), dtype='bool')
    for indx, vec in enumerate(errs):
        outxs[indx, :] = find_1d_pattern_err(vec*dmd_err_block_w)
    err_aoi = cp.einsum('eo,eu->eou', outxs, y_blocks_multi_cp_err).astype(cp.bool)
    err_aoi = err_aoi.reshape((num_frames, dmd_err_w, dmd_sig_h)).astype(cp.uint8)[None, ...].repeat(3, axis=0)
    err_aoi *= 255
    err_aoi = cp.transpose(err_aoi, (1, 2, 3, 0))

    imgs = cp.zeros((num_frames, dmd_w, dmd_h, 3), dtype=cp.uint8)
    imgs[dmd_sig_s_multi] = sig_aoi
    imgs[dmd_err_s_multi] = err_aoi

    # imgs_temp = imgs[:, dmd_w//2:, :, :][:, columns_indices_dmd, rows_indices_dmd, :].copy()
    # imgs = cp.transpose(imgs, (0, 2, 1, 3))
    # imgs[:, :, dmd_w//2:, :] = imgs_temp
    # imgs_temp = imgs[:, :, dmd_w//2:, :][:, columns_vert_indices_dmd, rows_vert_indices_dmd, :].copy()
    # imgs_temp = cp.transpose(imgs_temp, (0, 2, 1, 3))
    # imgs[:, :, dmd_w//2:, :] = imgs_temp

    imgs_temp = imgs[:, dmd_w//2:, :, :][:, columns_indices_dmd, rows_indices_dmd, :].copy()
    imgs_temp = imgs_temp[:, columns_vert_indices_dmd, rows_vert_indices_dmd, :].copy()
    imgs_temp = cp.transpose(imgs_temp, (0, 2, 1, 3))

    imgs = cp.transpose(imgs, (0, 2, 1, 3))
    imgs[:, :, dmd_w//2:, :] = imgs_temp

    imgs = cp.flip(imgs, axis=1)
    imgs = cp.flip(imgs, axis=2)

    return imgs


# def make_dmd_image(arr, err):
#
#     global n, m, k
#     global dmd_block_w, dmd_block_h
#     global dmd_err_block_w, dmd_err_block_h
#     global r, gpu_dmd_centers_x, gpu_dmd_centers_y
#     global r_err, gpu_dmd_err_centers_x, gpu_dmd_err_centers_y
#
#     arr = cp.array(arr*dmd_block_w)
#
#     if err.shape[0] == k:
#         err = np.repeat(err, 2, axis=0)
#         err[::2, :] *= (err[::2, :] < 0).astype(int)
#         err[1::2, :] *= (err[1::2, :] > 0).astype(int)
#         err = cp.array(np.abs(err)*dmd_err_block_w)
#     else:
#         err = cp.array(np.abs(err)*dmd_err_block_w)
#
#     img = cp.zeros((dmd_w, dmd_h), dtype='bool')
#
#     ##################
#
#     # noinspection PyTypeChecker
#     ws_local, ms_local, ns_local = cp.meshgrid(cp.arange(dmd_block_w+1), cp.arange(m), cp.arange(n),
#                                                indexing='ij')
#     target = arr[..., cp.newaxis].repeat(dmd_block_w + 1, axis=-1)
#     mask = (target > cp.arange(dmd_block_w + 1))
#     mask = mask.transpose(2, 1, 0)
#     mapped = cp.zeros((dmd_block_w + 1, dmd_sig_h, dmd_sig_w), dtype='bool')
#     dmd_xcc = gpu_dmd_centers_x - r[:, cp.newaxis, cp.newaxis]
#     dmd_ycc = gpu_dmd_centers_y[cp.newaxis, ...].repeat(dmd_block_w+1, axis=0)
#     for jj in range(dmd_block_h // 2):
#         mapped[ws_local, dmd_ycc - jj, dmd_xcc] = mask
#         mapped[ws_local, dmd_ycc + jj + 1, dmd_xcc] = mask
#     out = mapped.sum(axis=0)
#     out = out.T
#
#     img[dmd_sig_s] = out.copy()
#
#     ##################
#
#     # noinspection PyTypeChecker
#     ws_local, ms_local, ks_local = cp.meshgrid(cp.arange(dmd_err_block_w+1), cp.arange(m), cp.arange(2*k),
#                                                indexing='ij')
#     target = err[..., cp.newaxis].repeat(dmd_err_block_w + 1, axis=-1)
#     mask = (target > cp.arange(dmd_err_block_w + 1))
#     mask = mask.transpose(2, 1, 0)
#     mapped = cp.zeros((dmd_err_block_w + 1, dmd_sig_h, dmd_sig_w), dtype='bool')
#     dmd_xcc = gpu_dmd_err_centers_x - r_err[:, cp.newaxis, cp.newaxis]
#     dmd_ycc = gpu_dmd_err_centers_y[cp.newaxis, ...].repeat(dmd_err_block_w+1, axis=0)
#     for jj in range(dmd_block_h // 2):
#         mapped[ws_local, dmd_ycc - jj, dmd_xcc] = mask
#         mapped[ws_local, dmd_ycc + jj + 1, dmd_xcc] = mask
#     out = mapped.sum(axis=0)
#     out = out.T
#
#     dmd_angle = -0.5
#     out = ndimage.rotate(out.get(), dmd_angle, reshape=False)
#     out = cp.array(out)
#
#     img[dmd_err_s] = out.copy()
#
#     ##################
#
#     img = img.T
#     img = cp.flip(img, 1).astype(cp.uint8)
#
#     img *= 255
#
#     tilt = 18
#     for ii in range(img.shape[0]):
#         img[ii, :img.shape[1]//2] = cp.roll(img[ii, :img.shape[1]//2], ii//tilt - 10)
#     for ii in range(img.shape[1]//2):
#         img[:, ii] = cp.roll(img[:, ii], 5)
#
#     return img.astype(cp.uint8)
