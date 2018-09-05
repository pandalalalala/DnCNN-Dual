import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    normScale = float(1/(data.max()-data.min()))
    x = data * normScale
    return x#/255.

def Im2Patch(img_A, win, winscale=4, stride=1):
    k = 0
    endc = img_A.shape[0]
    endw = img_A.shape[1]
    endh = img_A.shape[2]
    #print(endc, endw, endh)
    col_n = (endw - stride*2 - win)//stride
    row_n = (endh - stride*2 - win)//stride
    
    TotalPatNum = row_n * col_n

    #patch = img_A[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    #TotalPatNum = int(patch.shape[1] * patch.shape[2])
    Y = np.zeros([endc, win, win, TotalPatNum], np.float32)

    for i in range(col_n):
        for j in range(row_n):
            patch = img_A[:,stride*(i+1):stride*(i+1) + win,stride*(1+j):stride*(1+j) + win]
            #Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            Y[:, :,:,k] = patch
            k = k + 1
    return Y#.reshape([endc, win, win, TotalPatNum])

def num_of_patch(img_A, scales, win, stride=1):
    TotalPatNum = 0
    for k in range(len(scales)):
        Img_A = cv2.resize(img_A, (int(img_A.shape[0]*scales[k]), int(img_A.shape[1]*scales[k])), interpolation=cv2.INTER_CUBIC)
        endw = Img_A.shape[0]
        endh = Img_A.shape[1]
        col_n = (endw - stride*2 - win)//stride
        row_n = (endh - stride*2 - win)//stride
        #col_n = (endw - stride*2)//win
        #row_n = (endh - stride*2)//win
        TotalPatNum += row_n * col_n
    return TotalPatNum

def prepare_train_data(data_path_A, data_path_B, patch_size_dn=15, patch_size_sr=60, stride=10, aug_times=1, if_reseize=True):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files_A = glob.glob(os.path.join('datasets', data_path_A, '*.*'))
    files_A.sort()

    files_B = glob.glob(os.path.join('datasets', data_path_B, '*.*'))
    files_B.sort()

    # assume all images in a single set have the same size
    lenA = len(files_A)
    lenScale = len(scales)
    lenPatch = num_of_patch(cv2.imread(files_A[0]), scales=scales, win=patch_size_sr, stride=stride*4)
    h_a, w_a, _ = cv2.imread(files_A[0]).shape
    h_b, w_b, _ = cv2.imread(files_B[0]).shape
    dataLength = (aug_times) * lenA * lenPatch
    print(lenPatch)
    data_dn = np.empty(shape=(2,dataLength, patch_size_dn, patch_size_dn))
    data_sr = np.empty(shape=(1,dataLength, patch_size_sr, patch_size_sr))


    train_num = 0
    for i in range(lenA):
        img_A = cv2.imread(files_A[i])
        img_B = cv2.imread(files_B[i])
        
        if if_reseize == True:
            img_L = cv2.resize(img_A, (h_b, w_b), interpolation=cv2.INTER_CUBIC)

        for k in range(lenScale):
            Img_A = cv2.resize(img_A, (int(h_a*scales[k]), int(w_a*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_A = np.expand_dims(Img_A[:,:,0].copy(), 0)
            Img_A = np.float32(normalize(Img_A))

            Img_L = cv2.resize(img_L, (int(h_b*scales[k]), int(w_b*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_L = np.expand_dims(Img_L[:,:,0].copy(), 0)
            Img_L = np.float32(normalize(Img_L))

            Img_B = cv2.resize(img_B, (int(h_b*scales[k]), int(w_b*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img_B = np.expand_dims(Img_B[:,:,0].copy(), 0)
            Img_B = np.float32(normalize(Img_B))

            patches_A = Im2Patch(Img_A, win=patch_size_sr, winscale=4, stride=stride*4)
            patches_L = Im2Patch(Img_L, win=patch_size_dn, winscale=1, stride=stride)
            patches_B = Im2Patch(Img_B, win=patch_size_dn, winscale=1, stride=stride)

            #print("file: %s scale %.1f # samples: %d" % (files_A[i], scales[k], patches_A.shape[3]*aug_times))
            #print("file: %s scale %.1f # samples: %d" % (files_B[i], scales[k], patches_A.shape[3]*aug_times))
            for n in range(patches_A.shape[3]):
                data_A = patches_A[:,:,:,n].copy()
                data_L = patches_L[:,:,:,n].copy()
                data_B = patches_B[:,:,:,n].copy()
                
                data_LB= [data_L, data_B]
                data_dn[:, train_num,:,:] = data_LB
                data_sr[:, train_num,:,:] = [data_A]
                #data_sr = np.append(data_sr, [data_A], axis=1)
                #data_dn = np.append(data_dn, data_LB, axis=1)

                train_num += 1
                for m in range(aug_times-1):
                    rand = np.random.randint(1,8)
                    data_aug_A = data_augmentation(data_A, rand)
                    data_aug_L = data_augmentation(data_L, rand)
                    data_aug_B = data_augmentation(data_B, rand)
                    data_aug_LB= [data_aug_L, data_aug_B]

                    data_dn[:, train_num,:,:] = data_aug_LB
                    data_sr[:, train_num,:,:] = [data_aug_A]
                    #data_dn = np.append(data_dn, data_aug_LB, axis=1)
                    #data_sr = np.append(data_sr, [data_aug_A], axis=1)
                    train_num += 1
    print('training set, # samples %d\n' % train_num)
    return data_sr, data_dn

def prepare_val_data(data_path_A, data_path_B, if_reseize=True):
    # val
    print('\nprocess validation data')
    files_A = glob.glob(os.path.join('datasets', data_path_A, '*.*'))
    files_A.sort()

    files_B = glob.glob(os.path.join('datasets', data_path_B, '*.*'))
    files_B.sort()

    data_n = np.empty(shape=[1, 0, 64, 64])
    data_s = np.empty(shape=[1, 0, 256, 256])
    val_num = 0
    for i in range(len(files_A)):
        print("file: %s" % files_A[i])
        img_A = cv2.imread(files_A[i])
        img_A = np.expand_dims(img_A[:,:,0], 0)
        img_A = np.float32(normalize(img_A))
        
        data_s = np.append(data_s, [img_A], axis=1)

        img_B = cv2.imread(files_B[i])
        if if_reseize == True:
            h_b, w_b, _ = img_B.shape
            img_L = cv2.resize(img_A, (h_b, w_b), interpolation=cv2.INTER_CUBIC)
        img_L = np.expand_dims(img_L[:,:,0], 0)
        img_L = np.float32(normalize(img_L))
        
        img_B = np.expand_dims(img_B[:,:,0], 0)
        img_B = np.float32(normalize(img_B))
        data_n = np.append(data_n, [img_B], axis=1)
        val_num += 1

    print('val set, # samples %d\n' % val_num)
    return data_s, data_n


class Dataset(udata.Dataset):
    def __init__(self, train, data_path_A, data_path_B, data_path_val_A, data_path_val_B,  patch_size_dn=30, patch_size_sr=50, stride=10, aug_times=2, if_reseize=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            self.data_S, self.data_N = prepare_train_data(data_path_A, data_path_B, patch_size_dn, patch_size_sr, stride, aug_times, if_reseize)

        else:
            self.data_S, self.data_N = prepare_val_data(data_path_val_A, data_path_val_B, if_reseize)
        self.num = self.data_S.shape[0]
        self.A = data_path_A
        self.B = data_path_B
        self.valA = data_path_val_A
        self.valB = data_path_val_B
        self.patch_size_dn = patch_size_dn
        self.patch_size_sr = patch_size_sr
        self.stride = stride
        self.aug_times = aug_times
        self.if_reseize = if_reseize
        
        
    def __len__(self):
        return self.num
    def __getitem__(self, index):
        if self.train:
            one_data_sr = self.data_S[:,index,:,:]
            one_data_dn = self.data_N[:,index,:,:]
        else:
            one_data_sr = self.data_S[:,index,:,:]
            one_data_dn = self.data_N[:,index,:,:]

        return torch.Tensor(one_data_sr), torch.Tensor(one_data_dn)

