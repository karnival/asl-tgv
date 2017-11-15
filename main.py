import nibabel as nib
import numpy as np


def update_step(P, Q, p, r, q, s, u_c, u_l, v, w, ubar_c, ubar_l, vbar, wbar):
    P_new = Pfun(lambd, P + sigma*(stack(ubar_c) - u_c_d))
    Q_new = Pfun(lambd, Q + sigma*(stack(ubar_l) - u_l_d))

    p_new = Pfun(gamma1 * alpha1, p + sigma*(grad(ubar_l) - vbar))
    r_new = Pfun(gamma2 * alpha1, r + sigma*(grad(ubar_c - ubar_l) - wbar))
    q_new = Pfun(gamma1 * alpha0, q + sigma*epsilon(vbar))
    s_new = Pfun(gamma2 * alpha0, s + sigma*epsilon(wbar))

    u_c_new = u_c + tau * (div(r_new) - stack(P_new))
    u_l_new = u_l + tau * (div(p_new - r_new) - stack(Q_new))

    v_new = v + tau * (p_new + div(q_new))
    w_new = w + tau * (r_new + div(s_new))

    ubar_c_new = 2*u_c_new - u_c
    ubar_l_new = 2*u_l_new - u_l

    vbar_new = 2*v_new - v
    wbar_new = 2*w_new - w


def stack(data):
    # Replicate data in 4th dimension to be the same size as the measured ASL.
    to_ret = np.tile(data, (1, 1, 1, datapoints))
    return to_ret


def epsilon(arg):
    D1 = backward_diff(arg, 0)
    D2 = backward_diff(arg, 1)
    D3 = backward_diff(arg, 2)

    M11 = D1[0]
    M22 = D2[1]
    M33 = D3[2]

    M12 = (D1[1] + D2[0]) / 2
    M13 = (D1[2] + D3[0]) / 2
    M23 = (D2[2] + D3[1]) / 2

    M = [[M11, M12, M13],
         [M12, M22, M23],
         [M13, M23, M33]]

    return M


def grad(arg):
    g0 = forward_diff(arg, 0)
    g1 = forward_diff(arg, 1)
    g2 = forward_diff(arg, 2)

    G = [g0, g1, g2]

    return G


def div(arg):
    pass


def forward_diff(arg, dimn):
    pass


def backward_diff(arg, dimn):
    pass


def Pfun(a, b):
    for i in range(np.shape(b, 1)):
        for j in range(np.shape(b, 1)):
            for k in range(np.shape(b, 1)):
                to_ret[i,j,k] = b[i,j,k] / max(1, np.norm(b[i,j,:], 2) / a)

    return to_ret


def main():
    filename = 'tmp_asl.nii.gz'

    data = nib.load(filename).get_data()


if __name__ == '__main__':
    main()
