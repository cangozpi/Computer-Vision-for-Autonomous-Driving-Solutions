import numpy as np
import cv2
import matplotlib.pyplot as plt

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write2ply(pts, fp):
    pts = pts.reshape(-1, 3)
    colors = np.zeros(pts.shape)
    pts_colors = np.hstack([pts, colors])
    with open(fp, 'w') as f:
        f.write(ply_header % dict(vert_num=len(pts_colors)))
        np.savetxt(f, pts_colors, '%f %f %f %d %d %d')
