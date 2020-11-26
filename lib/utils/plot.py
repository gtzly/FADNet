import cv2
from enum import Enum

from lib.utils.math import *

class cv_colors(Enum):
    RED = (0,0,255)
    GREEN = (0,255,0)
    BLUE = (255,0,0)
    PURPLE = (247,44,200)
    ORANGE = (44,162,247)
    MINT = (239,255,66)
    YELLOW = (2,255,250)

def constraint_to_color(constraint_idx):
    return {
        0 : cv_colors.PURPLE.value, #left
        1 : cv_colors.ORANGE.value, #top
        2 : cv_colors.MINT.value, #right
        3 : cv_colors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = (int(corner1_2d[0]), int(corner1_2d[1]))
    pt2 = (int(corner1_2d[0]), int(corner2_2d[1]))
    pt3 = (int(corner2_2d[0]), int(corner2_2d[1]))
    pt4 = (int(corner2_2d[0]), int(corner1_2d[1]))

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(pt, cam_to_img):
    point = np.array(pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, R0_rect), Tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point


def plot_3d_box(img, cam_to_img, ry, dimension, center):

    linewidth=2
    R = rotation_matrix(ry)
    corners = create_corners(dimension, location=center, R=R)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[2][0],box_3d[2][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), (box_3d[4][0],box_3d[4][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), (box_3d[6][0],box_3d[6][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[3][0],box_3d[3][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), (box_3d[5][0],box_3d[5][1]), cv_colors.ORANGE.value, linewidth,lineType=cv2.LINE_AA)

    for i in range(0,7,2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), (box_3d[i+1][0],box_3d[i+1][1]), cv_colors.ORANGE.value, linewidth)

    contours = np.array([[box_3d[0][0], box_3d[0][1]], [box_3d[1][0], box_3d[1][1]],
                         [box_3d[3][0], box_3d[3][1]], [box_3d[2][0], box_3d[2][1]]])

    contours=contours.astype(np.int64)
    zeros = np.zeros((img.shape), dtype=np.uint8)
    ret=cv2.fillPoly(zeros, pts=[contours], color=cv_colors.ORANGE.value)

    return ret

def plot_2d_box(img, box_2d):
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, cv_colors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, cv_colors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, cv_colors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, cv_colors.BLUE.value, 2)

def plot_bird_view(preds,world_size=64,out_size=768):
    bird_view = np.ones((out_size, out_size, 3), dtype=np.uint8) * 230

    for pred in preds:
        cls_idx,dims,locs,r_y=pred
        R = rotation_matrix(r_y)
        corners = create_corners(dims, location=locs, R=R)
        pts = np.array([corners[i] for i in (0, 1, 5, 4)])[:, [0, 2]]
        pts[:,0] += world_size / 2
        pts[:,1] = world_size - pts[:,1]
        pts = (pts * out_size / world_size).astype(np.int32)
        cv2.polylines(
            bird_view, [pts.reshape(-1, 1, 2)], True,
            (0,0,255), 2, lineType=cv2.LINE_AA)
        cv2.line(bird_view,(pts[0][0],pts[0][1]),(pts[1][0],pts[1][1]),
                 (0,0,255), 4, lineType=cv2.LINE_AA)
    return bird_view
