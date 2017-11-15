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
    pass


def epsilon(arg):
    pass


def grad(arg):
    pass


def div(arg):
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
