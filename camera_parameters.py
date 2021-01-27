import cv2
import numpy as np
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

fx = 786.356621482640
fy = 785.893340408213
cx = 628.382404554185
cy = 340.808726800420
k1 = -0.364895938951223
k2 = 0.205910153511996
k3 = -0.0835120929150047
p1 = -0.000288373571966727
p2 = 0.000262603352527099
camera_height = 0.6
# rotation_matrix = torch.eye(3)
rotation_vector = np.array([[-1.5035],
                            [0.03125],
                            [-0.06]])

rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
rotation_matrix = torch.FloatTensor(rotation_matrix)
# translation_vector = torch.FloatTensor([[-702.05089465,
#                                          607.54946787,
#                                          3048.32203209]]).t()
translation_vector = torch.FloatTensor([[0, 0, 0]]).t()

intrinsic_matrix = torch.FloatTensor(
    [[fx, 0, cx],
     [0, fy, cy],
     [0, 0, 1]])
extrinsics_matrix = torch.cat(
    (
        torch.cat((rotation_matrix, translation_vector), dim=1),
        torch.FloatTensor([[0, 0, 0, 1]])
    ), 0)
dist_coeffs = np.array((k1, k2, p1, p2, k3))

intrinsic_matrix = intrinsic_matrix.to(device)
extrinsics_matrix = extrinsics_matrix.to(device)
inv_extrinsics_matrix = extrinsics_matrix.inverse()


# u = torch.FloatTensor([1, 2, 3, 4, 5])
# v = torch.FloatTensor([6, 7, 8, 9, 10])
# depth_vector = torch.FloatTensor([11, 12, 13, 14, 15])
# ones = torch.ones(depth_vector.size())
# P_cam = torch.stack((u, v, depth_vector, ones), dim=0).to(device)
# P_w = torch.mm(inv_extrinsics_matrix, P_cam)

# np.savetxt('P_cam.txt', P_cam.cpu().numpy())
# np.savetxt('P_w.txt', P_w.cpu().numpy())
