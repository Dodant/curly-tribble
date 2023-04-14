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


def get_neighbor_matrix(image, T, ksize):
    height, width = image.shape
    neighborhood = sparse.lil_matrix((height * width, height * width)) # 337,500 * 337,500
    numbering = 0
    # Laplacian 
    lap = cv2.Laplacian(image, cv2.CV_8U, ksize=ksize)
    norm = np.linalg.norm(lap)
    norm_lap = lap / norm
    

    for i in range(height):
        for j in range(width):
            coords, values = get_neighbor_pixels(norm_lap, (i, j), T)
            for (x, y), weight in zip(coords, values):
                neighborhood[numbering, x * width + y] = weight
            numbering += 1
    return neighborhood


def get_scribbles(scribbles_img):
    return np.sum(cv2.imread(scribbles_img), axis=2).reshape(-1)

# I - w
def get_identity_weights(neighbor, scribbles_flat):
    for i in range(neighbor.shape[0]):
        if scribbles_flat[i] != 0: neighbor[i, :] = 0
    return sparse.identity(neighbor.shape[0]) - neighbor


def least_sq(i_minus_weight, scribbles_flat, h, w):
    x_back = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==1,1,0))
    x_fore = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==2,1,0))
    
    n = np.stack([x_back[0], x_fore[0]], axis=0).argmax(axis=0)
    return np.reshape(n, (h, w))


def get_ground_truth(gt_img):
    return np.sum(cv2.imread(gt_img, cv2.COLOR_BGR2RGB), axis=2)


def get_iou_score(gt, spm):
    intersection, union = np.logical_and(gt, spm), np.logical_or(gt, spm)
    score = np.sum(intersection) * 100 / np.sum(union)
    print(f'Intersection = {np.sum(intersection)}')
    print(f'Union = {np.sum(union)}')
    print(f'IoU score = {score}')
    return intersection, union, score


def make_plot(gt, spm, intersection, union):
    plt.title('Ground Truth')
    plt.imshow(gt)
    plt.savefig('lap-gt.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title('Output')
    plt.imshow(spm)
    plt.savefig('lap-output.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title('Intersection')
    plt.imshow(intersection)
    plt.savefig('lap-intersection.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title('Union')
    plt.imshow(union)
    plt.savefig('lap-union.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    print('Done')
    

def all_in_one(original_img, scribble_img, gt_img, T, ksize):
    img = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    neighbor = get_neighbor_matrix(img, T, ksize)
    scribbles_flat = get_scribbles(scribble_img)
    i_minus_weight = get_identity_weights(neighbor, scribbles_flat)
    spm = least_sq(i_minus_weight, scribbles_flat, height, width)
    gt = get_ground_truth(gt_img)
    intersection, union, _ = get_iou_score(gt, spm)
    make_plot(gt, spm, intersection, union)


original = 'dataset/Emily-In-Paris-gray.png'
scribble = 'dataset/Emily-In-Paris-scribbles.png'
gt_img = 'dataset/Emily-In-Paris-gt.png'
T, ksize = 9, 7

all_in_one(original, scribble, gt_img, T, ksize)