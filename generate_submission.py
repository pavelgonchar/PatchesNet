import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import scipy.misc as misc
import os
from u_net import get_unet_128, get_unet_256, get_unet_256_andreas, get_unet_1024
import copy
import mmap
from tqdm import tqdm
import sys
import random

mod = int(sys.argv[1])
# patch size
N = 512
input_size = 512
batch_size = 16*2
threshold = 0.5
#max_lines_count = 1000

model = get_unet_1024((input_size, input_size, 3))
#model.load_weights(filepath='../PatchesNet-binaries/weights/patchesnet_unet1024_noaug_sym_pad_skipcon_384_train_99.6248_val-99.5609', by_name=True)
model.load_weights(filepath='../PatchesNet-binaries/weights/patchesnet_unet1024_noaug_sym_pad_skipcon_512_continue_train-99.7048_val-99.6293', by_name=True)
f = open("../submit/remove_very_small_holes-11_PN512_epoch3.csv" + '.' + str(mod), "w")


def get_num_lines(file_path):
    fp = open(file_path, "r+")
    buf = mmap.mmap(fp.fileno(), 0)
    lines = 0
    while buf.readline():
        lines += 1
    return lines

def dice(im1, im2):
    im1 = np.asarray(im1).astype(np.bool)
    im2 = np.asarray(im2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())


def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)

def run_length_encode(mask):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    inds = mask.flatten()
    runs = np.where(inds[1:] != inds[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    rle = ' '.join([str(r) for r in runs])
    return rle

dices = []

#print f.write('img,rle_mask\n')
lineno = 0
with open("../submit/remove_very_small_holes-11.csv") as csv:
    for string in tqdm(csv, total=get_num_lines("../submit/remove_very_small_holes-11.csv")):
        string = string.strip()

        if 'rle' in string:
            continue

        lineno = lineno + 1
        if lineno % 4 != mod:
            continue

        #if len(dices) == max_lines_count:
        #    break

        original_image = cv2.imread('/data/pavel/carv/test_hq/%s' % string.split(',')[0])
        img = original_image
        #img = np.pad(img, ((N // 2, N // 2), (N // 2, N // 2), (0, 0)), 'symmetric')

        original_mask = rle_decode(string.split(',')[1], (1280, 1918))
        #original_mask = np.pad(original_mask, ((N // 2, N // 2), (N // 2, N // 2)), 'symmetric')

        border = np.abs(np.gradient(original_mask)[1]) + np.abs(np.gradient(original_mask)[0])
        border = np.select([border == 0.5, border != 0.5], [1.0, border])

        height, width = original_mask.shape

        patches_img = []
        patches_loc = []

        i = 0
        for x, y in zip(np.nonzero(border)[0], np.nonzero(border)[1]):
            #if i % 50 == 0 and x - N // 2 >= 0 and y - N // 2 >= 0 and x + N // 2 < img.shape[0] and y + N // 2 < img.shape[1]:
            if i % 50 == 0:

                x1 = x - N // 2
                x2 = x + N // 2
                y1 = y - N // 2
                y2 = y + N // 2
                
                if x1<0:
                    x2 = N
                    x1 = 0
                    
                if x2>img.shape[0]:
                    x1 = img.shape[0] - N
                    x2 = img.shape[0]

                if y1<0:
                    y2 = N
                    y1 = 0
                    
                if y2>img.shape[1]:
                    y1 = img.shape[1] - N
                    y2 = img.shape[1]
                
                patches_img.append(img[x1:x2,y1:y2, :])
                patches_loc.append([x1, x2, y1, y2])

                # cv2.rectangle(img,(y-N//2-1,x-N//2-1),(y+N//2,x+N//2),(255,255,0),1)
                #if N != input_size:
                #    patches_img.append(cv2.resize(img[x - N // 2:x + N // 2, y - N // 2:y + N // 2,:], (input_size, input_size), interpolation=cv2.INTER_CUBIC))
                #else:
                #    patches_img.append(img[x - N // 2:x + N // 2, y - N // 2:y + N // 2,:])
                    
                #patches_loc.append([x - N // 2, x + N // 2, y - N // 2, y + N // 2])
            i = i + 1
        is_flipped = []
        def test_generator():
            while True:
                for start in range(0, len(patches_img), batch_size):
                    x_batch = []
                    end = min(start + batch_size, len(patches_img))
                    test_batch = copy.deepcopy(patches_img)[start:end]                   
                    for im in test_batch:
                        flip = bool(random.getrandbits(1))
                        if flip:
                            im  = cv2.flip(im, 1)
                            x_batch.append(im)
                        else:
                            x_batch.append(im)
                        is_flipped.append(flip)
                    x_batch = np.array(x_batch, np.float32) / 255
                    yield x_batch
                    
        preds = model.predict_generator(generator=test_generator(),
                                    steps=np.ceil(float(len(patches_img)) / float(batch_size)))
        preds = np.squeeze(preds, axis=3)

        weights = np.zeros(original_mask.shape, dtype=np.float32)
        mask = np.zeros(original_mask.shape, dtype=np.float32)

        original_mask_patches = np.zeros_like(original_mask)
        original_mask_without_patches = copy.deepcopy(original_mask)
        j = 0
        for pred, patch, loc in zip(preds, patches_img, patches_loc):
            if is_flipped[j]:
                pred = cv2.flip(pred, 1)
            j = j + 1
            if N != input_size:
                mask[loc[0]:loc[1], loc[2]:loc[3]] += cv2.resize(pred, (N, N))
                weights[loc[0]:loc[1], loc[2]:loc[3]] += 1.0
                original_mask_patches[loc[0]:loc[1], loc[2]:loc[3]] = original_mask[loc[0]:loc[1], loc[2]:loc[3]]
                original_mask_without_patches[loc[0]:loc[1], loc[2]:loc[3]] = np.zeros_like(original_mask[loc[0]:loc[1], loc[2]:loc[3]])
            else:
                mask[loc[0]:loc[1], loc[2]:loc[3]] += pred
                weights[loc[0]:loc[1], loc[2]:loc[3]] += 1.0
                original_mask_patches[loc[0]:loc[1], loc[2]:loc[3]] = original_mask[loc[0]:loc[1], loc[2]:loc[3]]
                original_mask_without_patches[loc[0]:loc[1], loc[2]:loc[3]] = np.zeros_like(original_mask[loc[0]:loc[1], loc[2]:loc[3]])
            # break
        mask = mask/weights

        # unpad
        #mask = mask[N//2:-N//2, N//2:-N//2]
        mask = mask > threshold
        mask = np.select([mask == True, mask == False], [np.array(255, dtype=np.uint8), np.array(0, dtype=np.uint8)])

        #original_mask_patches = original_mask_patches[N//2:-N//2, N//2:-N//2]
        #original_mask_without_patches = original_mask_without_patches[N//2:-N//2, N//2:-N//2]

        # plt.figure(figsize=(25, 25))
        # plt.imshow(mask)
        # plt.show()


        result_mask = (original_mask_without_patches+mask/255)
        rle = run_length_encode(result_mask)

        f.write(string.split(',')[0] + ', ' +  rle + '\n')
        #dices.append(dice(original_mask_patches+original_mask_without_patches, original_mask_without_patches+mask/255))
        #break

#print "DICE Mean:", np.array(dices).mean()






