import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import linalg

def get_neighbor_pixels(lap, coord, T):
    x, y = coord
    s_coord, s_value = [], []
    
    for i in range(-T, T+1):
        for j in range(-T, T+1):
            if i == j == 0: continue
            try: 
                if x + i < 0 or y + j < 0: continue
                s = lap[x+i][y+j] 
                s_coord.append((x+i, y+j))
                s_value.append(s)
            except IndexError: continue     
    return s_coord, s_value


def get_neighbor_matrix(image, T):
    height, width = image.shape
    neighborhood = sparse.lil_matrix((height * width, height * width)) # 337,500 * 337,500
    lap = cv2.Laplacian(image, cv2.CV_8U, ksize=5)
    numbering = 0

    for i in range(height):
        for j in range(width):
            r = lap[i][j]
            coords, values = get_neighbor_pixels(lap, (i, j), T)
            for (x, y), weight in zip(coords, values):
                neighborhood[numbering, x * width + y] = weight
            numbering += 1
    return neighborhood


def get_scribbles(scribbles_img):
    scribbles = cv2.imread(scribbles_img)
    return np.sum(scribbles, axis=2).reshape(-1)

# I - w
def get_identity_weights(neighbor, scribbles_flat, height, width):
    identity_matrix = sparse.identity(height * width)
    for i in range(neighbor.shape[0]):
        if scribbles_flat[i] != 0: neighbor[i, :] = 0
    return identity_matrix - neighbor


def least_sq(i_minus_weight, scribbles_flat, h, w):
    print('lstsq start')
    x_back = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==1,1,0))
    x_fore = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==2,1,0))
    print('lstsq finish')
    
    n = np.stack([x_back[0], x_fore[0]], axis=0)
    c = n.argmax(axis=0)
    return np.reshape(c, (h, w))


def get_ground_truth(gt_img):
    gt = cv2.imread(gt_img, cv2.COLOR_BGR2RGB)
    return np.sum(gt, axis=2)


def get_iou_score(gt, spm):
    intersection, union = np.logical_and(gt, spm), np.logical_or(gt, spm)
    score = np.sum(intersection) * 100 / np.sum(union)
    print('Intersection =', np.sum(intersection))
    print('Union =', np.sum(union))
    print('IoU score =', score)
    return intersection, union, score


def make_plot(T, gt, spm, intersection, union):
    plt.title(f'T={T} Ground Truth')
    plt.imshow(gt)
    plt.savefig(f't{T}-gt.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title(f'T={T} Output')
    plt.imshow(spm)
    plt.savefig(f't{T}-output.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title(f'T={T} Intersection')
    plt.imshow(intersection)
    plt.savefig(f't{T}-intersection.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title(f'T={T} Union')
    plt.imshow(union)
    plt.savefig(f't{T}-union.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    print('Done')
    

def all_in_one(original_img, scribble_img, gt_img, T):
    img = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    neighbor = get_neighbor_matrix(img, T)
    scribbles_flat = get_scribbles(scribble_img)
    i_minus_weight = get_identity_weights(neighbor, scribbles_flat, height, width)
    spm = least_sq(i_minus_weight, scribbles_flat, height, width)
    gt = get_ground_truth(gt_img)
    intersection, union, score = get_iou_score(gt, spm)
    make_plot(T, gt, spm, intersection, union)


original = 'dataset/Emily-In-Paris-gray.png'
scribble = 'dataset/Emily-In-Paris-scribbles.png'
gt_img = 'dataset/Emily-In-Paris-gt.png'


all_in_one(original, scribble, gt_img, 5)