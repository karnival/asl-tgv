import nibabel as nib
import numpy as np


def update_step(u_c_d, u_l_d, P, Q, p, r, q, s, u_c, u_l, v, w, ubar_c, ubar_l, vbar, wbar):
    lambd = 5
    alpha0 = 1
    alpha1 = 1.41
    gamma1 = 1
    gamma2 = 1.5

    # Not actually calculated what these have to be yet.
    sigma = 0.1
    tau = 0.1


    reps = np.size(u_c_d, 3)

    P_new = Pfun(lambd, P + sigma*(stack(ubar_c, reps) - u_c_d))
    Q_new = Pfun(lambd, Q + sigma*(stack(ubar_l, reps) - u_l_d))

    p_new = Pfun(gamma1 * alpha1, p + sigma*(grad(ubar_l) - vbar))
    r_new = Pfun(gamma2 * alpha1, r + sigma*(grad(ubar_c - ubar_l) - wbar))

    q_new = Pfun(gamma1 * alpha0, q + sigma*epsilon(vbar))
    s_new = Pfun(gamma2 * alpha0, s + sigma*epsilon(wbar))

    # Index P_new at 0th timepoint because this is conjugate of stack().
    u_c_new = u_c + tau * (div(r_new) - P_new[:,:,:,0])
    u_l_new = u_l + tau * (div(p_new - r_new) - Q_new[:,:,:,0])

    v_new = v + tau * (p_new + div(q_new))
    w_new = w + tau * (r_new + div(s_new))

    ubar_c_new = 2*u_c_new - u_c
    ubar_l_new = 2*u_l_new - u_l

    vbar_new = 2*v_new - v
    wbar_new = 2*w_new - w

    return P_new, Q_new, p_new, r_new, q_new, s_new, u_c_new, u_l_new, v_new,\
           w_new, ubar_c_new, ubar_l_new, vbar_new, wbar_new

def stack(data, datapoints):
    # Replicate data in 4th dimension to be the same size as the measured ASL.
    to_ret = np.expand_dims(data, axis=3)
    to_ret = np.repeat(to_ret, datapoints, axis=3)
    return to_ret


def epsilon(arg):
    D1 = backward_diff(arg, 0)
    D2 = backward_diff(arg, 1)
    D3 = backward_diff(arg, 2)

    M11 = D1[:,:,:,0]
    M22 = D2[:,:,:,1]
    M33 = D3[:,:,:,2]

    M12 = (D1[:,:,:,1] + D2[:,:,:,0]) / 2
    M13 = (D1[:,:,:,2] + D3[:,:,:,0]) / 2
    M23 = (D2[:,:,:,2] + D3[:,:,:,1]) / 2

    M = np.empty(np.shape(arg) + (3,))

    M[:,:,:,0,0] = M11
    M[:,:,:,0,1] = M12
    M[:,:,:,0,2] = M13
    M[:,:,:,1,0] = M12
    M[:,:,:,1,1] = M22
    M[:,:,:,1,2] = M23
    M[:,:,:,2,0] = M13
    M[:,:,:,2,1] = M23
    M[:,:,:,2,2] = M33

    return M


def grad(arg):
    g0 = forward_diff(arg, 0)
    g1 = forward_diff(arg, 1)
    g2 = forward_diff(arg, 2)

    new_G_shape = arg.shape + (3,)
    G = np.empty(new_G_shape)

    G[:,:,:,0] = g0
    G[:,:,:,1] = g1
    G[:,:,:,2] = g2

    return G


def div(arg):
    vec, tensor = False, False
    if np.squeeze(arg).ndim == 4: # Vector field: each (x,y,z) has a (v1,v2,v3).
        vec = True
    elif np.squeeze(arg).ndim == 5: # Tensor field: each (x,y,z) has a 3x3 mat.
        tensor = True

    if vec:
        D = backward_diff(arg[:,:,:,0], 0) + backward_diff(arg[:,:,:,1], 1) + backward_diff(arg[:,:,:,2], 2)
        return D
    elif tensor:
        Dx = forward_diff(arg, 0)
        Dy = forward_diff(arg, 1)
        Dz = forward_diff(arg, 2)

        # Unsure about this but seems plausible.
        ret_vec = np.empty(np.shape(arg)[0:-2] + (3,))
        ret_vec[:,:,:,0] = Dx[:,:,:,0,0] + Dy[:,:,:,0,1] + Dz[:,:,:,0,2]
        ret_vec[:,:,:,1] = Dx[:,:,:,0,1] + Dy[:,:,:,1,1] + Dz[:,:,:,2,1]
        ret_vec[:,:,:,2] = Dx[:,:,:,0,2] + Dy[:,:,:,1,2] + Dz[:,:,:,2,2]

        return ret_vec


def forward_diff(arg, dimn):
    if np.squeeze(arg).ndim == 3:
        diffed = np.diff(arg, axis=dimn)
        if dimn == 0:
            return np.concatenate([diffed, np.zeros([1, np.shape(diffed)[1], np.shape(diffed)[2]])], axis=0)
        elif dimn == 1:
            return np.concatenate([diffed, np.zeros([np.shape(diffed)[0], 1, np.shape(diffed)[2]])], axis=1)
        elif dimn == 2:
            return np.concatenate([diffed, np.zeros([np.shape(diffed)[0], np.shape(diffed)[1], 1])], axis=2)

    elif np.squeeze(arg).ndim == 4:
        diffed = np.empty(np.shape(arg))
        for i in range(arg.shape[3]):
            diffed[:,:,:,i] = forward_diff(arg[:,:,:,i], dimn)
        return diffed

    elif np.squeeze(arg).ndim == 5:
        diffed = np.empty(np.shape(arg))
        for i in range(arg.shape[3]):
            for j in range(arg.shape[4]):
                diffed[:,:,:,i,j] = forward_diff(arg[:,:,:,i,j], dimn)
        return diffed


def backward_diff(arg, dimn):
    new_arg = np.copy(arg)
    if np.squeeze(arg).ndim == 3:
        if dimn == 0:
            new_arg[-1,:,:] = 0
            diffed = np.diff(new_arg, axis=dimn)
            to_concat = np.expand_dims(new_arg[0,:,:], axis=0)
        elif dimn == 1:
            new_arg[:,-1,:] = 0
            diffed = np.diff(new_arg, axis=dimn)
            to_concat = np.expand_dims(new_arg[:,0,:], axis=1)
        elif dimn == 2:
            new_arg[:,:,-1] = 0
            diffed = np.diff(new_arg, axis=dimn)
            to_concat = np.expand_dims(new_arg[:,:,0], axis=2)

        to_ret = np.concatenate((to_concat, diffed), axis=dimn)
        return to_ret 

    elif np.squeeze(arg).ndim == 4:
        diffed = np.empty(np.shape(arg))
        for i in range(arg.shape[3]):
            diffed[:,:,:,i] = backward_diff(arg[:,:,:,i], dimn)

        return diffed


def Pfun(a, b):
    to_ret = np.empty(b.shape)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            for k in range(b.shape[2]):
                for l in range(b.shape[3]):
                    to_ret[i,j,k,l] = b[i,j,k,l] / max(1, np.linalg.norm(b[i,j,k,:], 2) / a)

    return to_ret


def main():
    filename = 'tmp_asl.nii.gz'

    data = nib.load(filename).get_data()

    u_l_d = data[:,:,:,0::2]
    u_c_d = data[:,:,:,1::2]

    u_l = np.nanmedian(u_l_d, 3)
    u_c = np.nanmedian(u_c_d, 3)

    P = u_c_d
    Q = u_c_d

    p = np.zeros(np.shape(u_l) + (3,))
    r = p

    q = np.zeros(np.shape(u_l) + (3, 3,))
    s = q

    ubar_c = u_c
    ubar_l = u_l

    v = p
    w = r

    vbar = v
    wbar = w

    update_step(u_c_d, u_l_d, P, Q, p, r, q, s, u_c, u_l, v, w, ubar_c, ubar_l, vbar, wbar)

if __name__ == '__main__':
    main()
