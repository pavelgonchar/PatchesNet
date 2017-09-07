import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import scipy.misc as misc
import os
from tqdm import tqdm
import errno

# patch size
N = 128
ROOT_DIR = '/data/pavel/carv'


def mkdir_p(path):
    """Utility function emulating mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

mkdir_p(ROOT_DIR + '/train_patches')
mkdir_p(ROOT_DIR + '/train_patches_masks')

ids = [os.path.basename(x) for x in glob.glob(ROOT_DIR + '/train_hq/*.jpg')]
ids = [x.split('.')[0] for x in ids]

for j in tqdm(range(len(ids))):
    mask = misc.imread(ROOT_DIR + '/train_masks/%s_mask.gif' %
                       ids[j], cv2.IMREAD_GRAYSCALE)[..., 0] / 255.0
    mask = np.pad(mask, ((N // 2, N // 2), (N // 2, N // 2)), 'symmetric')

    border = np.abs(np.gradient(mask)[1]) + np.abs(np.gradient(mask)[0])
    border = np.select([border == 0.5, border != 0.5], [1.0, border])

    img = cv2.imread(ROOT_DIR + '/train_hq/%s.jpg' % ids[j])
    img = np.pad(
        img, ((N // 2, N // 2), (N // 2, N // 2), (0, 0)), 'symmetric')

    height, width = mask.shape

    i = 0
    for x, y in zip(np.nonzero(border)[0], np.nonzero(border)[1]):
        if i % 50 == 0 and x - N // 2 >= 0 and y - N // 2 >= 0:
            misc.imsave(ROOT_DIR + '/train_patches/%s_%s.jpg' %
                        (ids[j], i), img[x - N // 2:x + N // 2, y - N // 2:y + N // 2, :])
            misc.imsave(ROOT_DIR + '/train_patches_masks/%s_%s.png' %
                        (ids[j], i), mask[x - N // 2:x + N // 2, y - N // 2:y + N // 2] * 255)
        i = i + 1
