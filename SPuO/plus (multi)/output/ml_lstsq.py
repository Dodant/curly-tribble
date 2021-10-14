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
    return weight_neighbor / np.sum(weight_neighbor)


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
    return np.sum(cv2.imread(scribbles_img), axis=2).reshape(-1)

# I - w
def get_identity_weights(neighbor, scribbles_flat):
    for i in range(neighbor.shape[0]):
        if scribbles_flat[i] != 0: neighbor[i, :] = 0
    return sparse.identity(neighbor.shape[0]) - neighbor


def least_sq(i_minus_weight, scribbles_flat, h, w):
    x_sky = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==1,1,0))
    x_bui = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==2,1,0))
    x_tre = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==3,1,0))
    x_hai = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==4,1,0))
    x_ski = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==5,1,0))
    x_pho = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==6,1,0))
    x_clo = linalg.lsqr(i_minus_weight, np.where(scribbles_flat==7,1,0))
    
    n = np.stack([x_sky[0], x_bui[0], x_tre[0], x_hai[0], x_ski[0], x_pho[0], x_clo[0]], axis=0).argmax(axis=0)
    return np.reshape(n, (h, w))


def get_ground_truth(gt_img):
    return np.sum(cv2.imread(gt_img, cv2.COLOR_BGR2RGB), axis=2)


def get_iou_score(gt, spm):
    intersections = []
    unions = []
    scores = []
    
    for i in range(7):
        gt_ind = np.where(gt==i+1,1,0)
        spm_ind = np.where(spm==i,1,0)
        intersection = np.logical_and(gt_ind, spm_ind)
        union = np.logical_or(gt_ind, spm_ind)
        score = np.sum(intersection) * 100 / np.sum(union)
        
        intersections.append(intersection)
        unions.append(union)
        scores.append(score)
        
        print('class =', i)
        print('Intersection =', np.sum(intersection))
        print('Union =', np.sum(union))
        print('IoU score =', score)
        print()
    
    print('mIoU = ', sum(scores) / len(scores))
    return intersections, unions, score


def make_plot(gt, spm, intersections, unions):
    plt.title(f'Multi-Label Ground Truth')
    plt.imshow(gt)
    plt.savefig(f'ml-gt.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    plt.title(f'Multi-Label Output')
    plt.imshow(spm)
    plt.savefig(f'ml-output.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    classes = ['Sky','Buildings','Tree','Hair','Skin','Phone','Clothes']
    
    for i in range(7):
        plt.title(f'{classes[i]} Ground Truth')
        plt.imshow(np.where(gt==i+1,1,0))
        plt.savefig(f'{classes[i]}-gt.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
        
        plt.title(f'{classes[i]} Output')
        plt.imshow(np.where(spm==i,1,0))
        plt.savefig(f'{classes[i]}-output.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')

        plt.title(f'{classes[i]} Intersection')
        plt.imshow(intersections[i])
        plt.savefig(f'{classes[i]}-intersection.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')

        plt.title(f'{classes[i]} Union')
        plt.imshow(unions[i])
        plt.savefig(f'{classes[i]}-union.png', facecolor='#eeeeee', edgecolor='blue', bbox_inches='tight')
    
    print('Done')
    

def all_in_one(original_img, scribble_img, gt_img, T):
    img = cv2.imread(original_img, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape
    
    neighbor = get_neighbor_matrix(img, T)
    scribbles_flat = get_scribbles(scribble_img)
    i_minus_weight = get_identity_weights(neighbor, scribbles_flat)
    
    spm = least_sq(i_minus_weight, scribbles_flat, height, width)
    gt = get_ground_truth(gt_img)
    intersections, unions, _ = get_iou_score(gt, spm)
    make_plot(gt, spm, intersections, unions)

original = 'dataset/Emily-In-Paris-gray.png'
scribble = 'dataset/Emily-In-Paris-scribbles-plus.png'
gt_img = 'dataset/Emily-In-Paris-gt-plus.png'

all_in_one(original, scribble, gt_img, 5)