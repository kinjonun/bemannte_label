import numpy as np
import torch


def get_matrix(e2g_trans, e2g_rot, inverse):
    matrix = torch.eye(4, dtype=torch.float64)
    if inverse:  # g2e_matrix
        matrix[:3, :3] = e2g_rot.T
        matrix[:3, 3] = -(e2g_rot.T @ e2g_trans)
    else:
        matrix[:3, :3] = e2g_rot
        matrix[:3, 3] = e2g_trans
    return matrix


# prev_e2g_trans = torch.tensor([5563.70302767, 2154.62443882,   74.84883636])
# prev_e2g_rot = torch.tensor([[-0.8230897399457003, -0.5674208453115417, 0.023597972414063292],
#  [0.5676674913822461, -0.823243282186288, 0.0049109635444995565],
#  [0.01664028917738084, 0.01743796550872371, 0.9997094668627529]], )
#
#
# curr_e2g_trans = torch.tensor([5500.46063008, 2197.37956826,   75.86030019])
# curr_e2g_rot = torch.tensor([[-0.8170720494325876, -0.5761794217348897, 0.020261786823462635],
#  [0.5763105767064203, -0.8172303841339343, 0.000786399780929716],
#  [0.016105440457949732, 0.01231962733000988, 0.9997943996493005]],)


prev_e2g_trans = torch.tensor([5563.7030, 2154.6244,   74.8488])
prev_e2g_rot = torch.tensor([[-0.8231, -0.5674,  0.0236],
        [ 0.5677, -0.8232,  0.0049],
        [ 0.0166,  0.0174,  0.9997]], )


curr_e2g_trans = torch.tensor([5500.4606, 2197.3796,   75.8603])
curr_e2g_rot = torch.tensor([[-8.1709e-01, -5.7615e-01,  2.0288e-02],
        [ 5.7628e-01, -8.1725e-01,  7.5886e-04],
        [ 1.6143e-02,  1.2312e-02,  9.9979e-01]],)

prev_e2g_matrix = get_matrix(prev_e2g_trans, prev_e2g_rot, False)
curr_g2e_matrix = get_matrix(curr_e2g_trans, curr_e2g_rot, True)
prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
print(prev2curr_matrix)

# tensor([[ 9.9994e-01, -1.0539e-02, -3.5024e-04, -7.6330e+01],
#         [ 1.0537e-02,  9.9993e-01, -5.2940e-03, -1.5109e+00],
#         [ 4.0601e-04,  5.2900e-03,  9.9999e-01,  2.3653e-01],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],


# tensor([[ 9.9997e-01, -1.0496e-02, -3.2140e-04, -7.6330e+01],
#         [ 1.0481e-02,  9.9988e-01, -5.2934e-03, -1.5081e+00],
#         [ 3.2827e-04,  5.2602e-03,  9.9997e-01,  2.3934e-01],
#         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]],