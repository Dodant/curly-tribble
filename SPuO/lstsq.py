import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse import linalg

def get_neighbor_pixels(image, coord, T):
    x, y = coord
    s_coord, s_value = [], []
    
    for i in range(-T, T+1):
        for j in range(-T, T+1):
            if i == j == 0: continue
            try: 
                if x + i < 0 or y + j < 0: continue
                s = image[x+i][y+j] 
                s_coord.append((x+i, y+j))
                s_value.append(s)
            except IndexError: continue     
    return s_coord, s_value


def weight_f(mean, var, r, s): # option 2
    eps = 1e-6
    return 1 + ((r-mean)*(s-mean))/(var+eps)


def get_weight(r, values):
    n_mean, n_var = np.mean(values), np.var(values)
    weight_neighbor = [weight_f(n_mean, n_var, r, i) for i in values]
    normalized_neighbor = weight_neighbor / np.sum(weight_neighbor)
    return normalized_neighbor


def get_neighbor_matrix(image, T):
    height, width = image.shape
    neighborhood = sparse.lil_matrix((height * width, height * width)) # 337,500 * 337,500
    numbering = 0

    for i in range(height):
        for j in range(width):
            r = image[i][j]
            coords, values = get_neighbor_pixels(image, (i, j), T)
            normalized_neighbor = get_weight(r, values)
            for (x, y), weight in zip(coords, normalized_neighbor):
                neighborhood[numbering, x * width + y] = weight
            numbering += 1
    return neighborhood


def get_scribbles(scribbles_img):
    scribbles = cv2.imread(scribbles_img)
    h, w, c = scribbles.shape
    scribbles_back, scribbles_fore = np.zeros((h * w,)), np.zeros((h * w,))
    numbering = 0

    for i in range(h):
        for j in range(w):
            if scribbles[i][j][2] == 1: scribbles_back[numbering] = 1
            elif scribbles[i][j][2] == 2: scribbles_fore[numbering] = 1
            numbering += 1
    return scribbles_back, scribbles_fore

# I - w
def get_identity_weights(neighbor, scribbles_back, scribbles_fore, height, width):
    identity_matrix = sparse.identity(height * width)
    for i in range(neighbor.shape[0]):
        if scribbles_back[i] == 1 or scribbles_fore[i] == 1:
            neighbor[i, :] = 0
    return identity_matrix - neighbor


def least_sq(i_minus_weight, scribbles_back, scribbles_fore, h, w):
    print('lstsq start')
    x_back = linalg.lsqr(i_minus_weight, scribbles_back)
    x_fore = linalg.lsqr(i_minus_weight, scribbles_fore)
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
    scribbles_back, scribbles_fore = get_scribbles(scribble_img)
    i_minus_weight = get_identity_weights(neighbor, scribbles_back, scribbles_fore, height, width)
    spm = least_sq(i_minus_weight, scribbles_back, scribbles_fore, height, width)
    gt = get_ground_truth(gt_img)
    intersection, union, score = get_iou_score(gt, spm)
    make_plot(T, gt, spm, intersection, union)


original = 'Emily-In-Paris-gray.png'
scribble = 'Emily-In-Paris-scribbles.png'
gt_img = 'Emily-In-Paris-gt.png'

for i in range(1, 6):
    all_in_one(original, scribble, gt_img, i)
    print(f'T = {i} Done ------------------')