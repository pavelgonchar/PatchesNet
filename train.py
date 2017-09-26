import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.models import load_model
from keras.engine.training import Model

from sklearn.model_selection import train_test_split
import glob
import u_net
from u_net import bce_dice_loss, dice_loss, dice_loss100, Scale
import os
import scipy.misc as misc
import random
from os.path import join
import errno
import itertools
import argparse
from multi_gpu import to_multi_gpu
import csv
from tps import tps
import re
from tqdm import tqdm
import copy


ROOT_DIR = '/data/pavel/carv'
WEIGHTS_DIR = '../PatchesNet-binaries/weights'

def mkdir_p(path):
    """Utility function emulating mkdir -p."""
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

mkdir_p(WEIGHTS_DIR)

parser = argparse.ArgumentParser()

# train 
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='Initial learning rate, e.g. -l 1e-2')
parser.add_argument('-lw', '--load-weights', type=str, help='load model weights (and continue training)')
parser.add_argument('-lm', '--load-model', type=str, help='load model (and continue training)')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-p', '--patch-size', type=int, default=384, help='Patch size, e.g -p 128')
parser.add_argument('-i', '--input-size', type=int, default=384, help='Network input size, e.g -i 256')
parser.add_argument('-ub', '--use-background', action='store_true', help='Use magic background as extra input to NN')
parser.add_argument('-uc', '--use-coarse', action='store_true', help='Use coarse mask as extra input to NN')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='Optimizer to use: adam, nadam, sgd, e.g. -o adam')
parser.add_argument('-at', '--augmentation-tps', action='store_true', help='TPS augmentation')
parser.add_argument('-af', '--augmentation-flips', action='store_true', help='Flips augmentation')
parser.add_argument('-s', '--suffix', type=str, default=None, help='Suffix for saving model name')
parser.add_argument('-m', '--model', type=str, default='dilated_unet', help='Use model, e.g. -m dilated_unet -m unet_256, unet_bg_256, largekernels')
parser.add_argument('-f', '--fractional-epoch', type=int, default=1, help='Reduce epoch steps by factor, e.g. -f 10 (after 10 epochs all samples would have been seen) ')

# test / submission 
parser.add_argument('-t', '--test', action='store_true', help='Test/Submit')
parser.add_argument('-tb', '--test-background', type=str, default=join(ROOT_DIR, 'test_background_hq_09970_pavel'), help='Magic backgrounds folder in PNG format for test, e.g. -tb /data/pavel/carv/test_backgroound_hq')
parser.add_argument('-tc', '--test-coarse', type=str, help='Coarse mask folder in PNG format for test, e.g. -tc /data/pavel/carv/09967_test')
parser.add_argument('-tf', '--test-folder', type=str, default=join(ROOT_DIR, 'test_hq'), help='Test folder e.g. -tc /data/pavel/carv/test_hq')
parser.add_argument('-tppi', '--test-patches-per-image', type=int, default=128, help='Patches per image (rounded to multiple of batch size)')
parser.add_argument('-tts', '--test-total-splits',  type=int, default=1, help='Only do the Xth car, e.g. -tts 12 (needs to work with -tcs)')
parser.add_argument('-tcs', '--test-current-split', type=int, default=0, help='Only do the Nth car out of Xth, e.g. -tts 12 -tcs 2')
parser.add_argument('-tsps', '--test-smart-patch-selection', action='store_true', help='Use a smarter way of selecting patches')

# common 
parser.add_argument('-g', '--gpus', type=int, default=1, help='Use GPUs, e.g -g 2')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training/test, e.g. -b 32')
parser.add_argument('-tcm', '--threshold-coarse', action='store_true', help='Threshold coarse mask (for training or eval)')

args = parser.parse_args()  

def preprocess_input_imagenet(img):
    return img.astype(np.float32) - np.float32([103.939, 116.779, 123.68])

# WARNING -> this would fail for 'largekernels' if LOADING MODEL (b/c args.model would be undefined)
# TODO: Fix if you plan to load models based on 'largekernels' architecture
preprocess_for_model = preprocess_input_imagenet if args.model == 'largekernels' else lambda x: x / 255.

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

PATCH_SIZE = args.patch_size
TRAIN_FOLDER_PATCHES = join(ROOT_DIR, 'train_patches_' + str(PATCH_SIZE))
TRAIN_FOLDER_MASKS   = join(ROOT_DIR, 'train_patches_masks_' + str(PATCH_SIZE))
BACKGROUNDS_FOLDER   = join(ROOT_DIR, 'train_background_hq_09970')
CSV_FILENAME         = join(ROOT_DIR, 'train_patches_' + str(PATCH_SIZE) + ".csv")
COARSE_FOLDER        = join(ROOT_DIR, '09970')

input_size = args.input_size
batch_size = args.batch_size 

if not args.test:
    all_files = glob.glob(join(TRAIN_FOLDER_PATCHES, '*_*.jpg'))
    ids = list(set([(x.split('/')[-1]).split('_')[0] for x in all_files]))
    ids.sort()

    if args.use_background or args.use_coarse:
        with open(CSV_FILENAME, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            patches_dict = {rows[0]:(int(rows[1]),int(rows[2])) for rows in reader}
            if args.use_background:
                background_dict = { }
                stats_dict      = { }
            if args.use_coarse:
                coarse_dict = { }

    ids_train_split, ids_valid_split = train_test_split(
        ids, test_size=0.1, random_state=13)

    ids_train_split = [os.path.basename(x).split('.')[0]
                         for x in all_files if os.path.basename(x).split('_')[0] in ids_train_split]
    ids_valid_split = [os.path.basename(x).split('.')[0]
                         for x in all_files if os.path.basename(x).split('_')[0] in ids_valid_split]

    ids_train_split += ids_valid_split

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

else:

    all_files = glob.glob(join(args.test_folder, '*_*.jpg'))
    ids = list(set([(x.split('/')[-1]).split('_')[0] for x in all_files]))
    ids.sort()
    print('Testing on {} samples'.format(len(ids)))

def randomShiftScaleRotate(image, mask,
                             shift_limit=(-0.0625, 0.0625),
                             scale_limit=(-0.1, 0.1),
                             rotate_limit=(-45, 45), aspect_limit=(0, 0),
                             borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])  # degree
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + \
            np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image, mat, (
                width, height), flags=cv2.INTER_CUBIC, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(
            mask, mat, (
                width, height), flags=cv2.INTER_CUBIC, borderMode=borderMode,
                                     borderValue=(
                                         0, 0,
                                         0,))
    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)

    return image, mask

def generator(ids, training = True):
    random.seed(13)
    while True:
        if training:
            random.shuffle(ids)
        for start in range(0, len(ids), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids))
            ids_batch = ids[start:end]
            i = 0
            for id in ids_batch:

                img = cv2.imread(join(TRAIN_FOLDER_PATCHES, '{}.jpg'.format(id)))
                if (input_size, input_size, 3) != img.shape:
                    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                
                mask = cv2.imread(join(TRAIN_FOLDER_MASKS, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
                if (input_size, input_size) != mask.shape:
                    mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

                if args.use_background:
                    background = None
                    car_id = id.split('_')[0]
                    if car_id in background_dict:
                        all_background = background_dict[car_id]
                    else:
                        all_background = cv2.imread(join(BACKGROUNDS_FOLDER, '{}.png'.format(car_id)))
                        all_background = np.pad(all_background, ((PATCH_SIZE // 2, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2), (0, 0)), 'symmetric')
                        background_dict[car_id] = all_background
                    patch_id = '{}.jpg'.format(id)
                    x,y = patches_dict[patch_id]
                    background = np.copy(all_background[y-PATCH_SIZE//2:y+PATCH_SIZE//2,x-PATCH_SIZE//2:x+PATCH_SIZE//2])
                    #cv2.imwrite("bb.png", background)

                    no_background_color = (255,0,255)
                    background_index = np.all(background != no_background_color, axis=-1)
                    selected_background = background[background_index]
                    background_l2 = np.expand_dims(255 - np.linalg.norm(background - img, axis=2) / np.sqrt(3.), axis=2)
                    background_mask = np.zeros((PATCH_SIZE, PATCH_SIZE,1), dtype=np.uint8)
                    background_mask[background_index] = 255
                    selected_background_l2 = background_l2[background_index]

                    #print(background_index.shape)
                    #print(selected_background_l2.shape)
                    if patch_id in stats_dict:
                        selected_background_mean,selected_background_std  = stats_dict[patch_id]
                    else:                
                        if selected_background.size > 0:
                            selected_background_mean = np.mean(selected_background_l2)
                            selected_background_std  = np.std(selected_background_l2)
                        else:
                            selected_background_mean = np.mean(img)
                            selected_background_std  = np.std(img)                    
                            stats_dict[patch_id] = (selected_background_mean, selected_background_std)

                    background_l2[~background_index] = \
                        np.random.normal(loc=selected_background_mean, scale=selected_background_std, size=(PATCH_SIZE**2-(selected_background.size//3),1))

                    img = np.concatenate([img, background_l2, background_mask], axis=2)

                if args.use_coarse:
                    car_view_id = id.split('_')[:2]
                    car_view_file = '{}_{}.png'.format(car_view_id[0], car_view_id[1])
                    if car_view_file in coarse_dict:
                        all_coarse = coarse_dict[car_view_file]
                    else:
                        all_coarse = cv2.imread(join(COARSE_FOLDER, car_view_file), cv2.IMREAD_GRAYSCALE)

                        if args.threshold_coarse:
                            all_coarse = 255 * np.rint(all_coarse/255.).astype(np.uint8)

                        all_coarse = np.pad(all_coarse, ((PATCH_SIZE // 2, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2)), 'symmetric')

                                              #coarse_dict[car_view_file] = all_coarse
                    patch_id = '{}.jpg'.format(id)
                    x,y = patches_dict[patch_id]
                    coarse = np.copy(all_coarse[y-PATCH_SIZE//2:y+PATCH_SIZE//2,x-PATCH_SIZE//2:x+PATCH_SIZE//2])
                    img = np.concatenate([img, np.expand_dims(coarse, axis=2)], axis=2)

                if training:
                    if args.augmentation_tps:
                        #print(img.shape, mask.shape)
                        img, mask = tps({'img': img, 'mask': mask, 'seed': random.randint(0,1000)})
                        img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                        mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

                    if args.augmentation_flips:
                        img, mask = randomHorizontalFlip(img, mask)

                mask = np.expand_dims(mask, axis=2)
                img = preprocess_for_model(img.astype(np.float32))
                x_batch.append(img)
                y_batch.append(mask)

                if img.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                    print(id)
            x_batch = np.array(x_batch, np.float32) 
            y_batch = np.array(y_batch, np.float32) / 255.
            yield x_batch, y_batch

def get_weighted_window(patch_size):
    squareX, squareY = np.meshgrid(
        np.arange(1, patch_size // 2 + 1, 1),
        np.arange(1, patch_size // 2 + 1, 1))
    grid = (squareX + squareY) // 2
    square = np.zeros((patch_size, patch_size), dtype=np.float32)
    square[0:patch_size // 2, 0:patch_size // 2] = grid
    square[patch_size // 2:, 0:patch_size // 2] = np.flip(grid, 0)
    square[0:patch_size // 2, patch_size // 2:] = np.flip(grid, 1)
    square[patch_size // 2:, patch_size // 2:] = patch_size // 2 + 1 - grid
    w = np.sqrt(np.sqrt(square / (patch_size // 2)))
    return w

def rle_encode(pixels):
    #pixels = pixels[:, :1918,:]
    pixels = pixels.ravel()
    np.rint(pixels, out=pixels)
    
    # We avoid issues with '1' at the start or end (at the corners of 
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask, 
    # so this should not harm the score.
    pixels[0]  = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs

def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def test_model(model, ids, X, CO, patches_per_image, batch_size, csv_filename, save_pngs_to_folder, input_channels):

    random.seed(13)

    def patches_generator(car_id, all_coarse_padded_batch, car_xy_flip_batch):
        
        patch_id = 0
        if X:
            all_background = cv2.imread(join(args.test_background, '{}.png'.format(car_id)))
            all_background_padded = np.pad(all_background, ((PATCH_SIZE // 2, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2), (0, 0)), 'symmetric')

        for idx in range(1,17):
            car_view_file = '{}_{:02d}'.format(car_id, idx) 

            img = cv2.imread(join(args.test_folder, car_view_file + '.jpg'))
            img_padded = np.pad(img, ((PATCH_SIZE // 2, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2), (0, 0)), 'symmetric')

            all_coarse = cv2.imread(join(args.test_coarse, car_view_file + '.png'), cv2.IMREAD_GRAYSCALE)
            if args.threshold_coarse:
                all_coarse = 255 * np.rint(all_coarse/255.).astype(np.uint8)
    
            all_coarse_padded_batch[idx-1, ...] = np.pad(all_coarse, ((PATCH_SIZE // 2, PATCH_SIZE // 2), (PATCH_SIZE // 2, PATCH_SIZE // 2)), 'symmetric')
    
            if args.threshold_coarse:
                all_coarse_mask = all_coarse
            else:
                all_coarse_mask = 255 * np.rint(all_coarse/255.).astype(np.uint8)

            border = np.abs(np.gradient(all_coarse_mask)[1]) + np.abs(np.gradient(all_coarse_mask)[0])                
            border = np.select([border == 0.5, border != 0.5], [1.0, border])

            edges = np.nonzero(border)

            seed = random.randint(0,1000)
            edges_x, edges_y = edges[0], edges[1]
            n_patches = batch_size * (patches_per_image // batch_size)

            if args.test_smart_patch_selection:
                edge_probs = []
                for y,x in zip(edges[0], edges[1]):
                    B = PATCH_SIZE // 32
                    x += PATCH_SIZE // 2 
                    y += PATCH_SIZE // 2 
                    edge_prob = 1. / (all_coarse_padded[idx-1, y-B//2:y+B//2,x-B//2:x+B//2].mean() + 1.)
                    edge_probs.append(edge_probs)

                random.seed(seed)
                edges_x = np.random.choice(edges_x, size = n_patches, replace=None, p=edge_probs)
                random.seed(seed)
                edges_y = np.random.choice(edges_y, size = n_patches, replace=None, p=edge_probs)
            else:
                random.seed(seed)
                random.shuffle(edges_x)
                random.seed(seed)
                random.shuffle(edges_y)

            edges = edges_x[: n_patches], edges_y[: n_patches]

            i = 0
            img_batch = np.empty((batch_size, input_size, input_size, input_channels), dtype=np.float32)

            xy_batch = []

            for y,x in zip(edges[0], edges[1]):
                x = x + PATCH_SIZE // 2 
                y = y + PATCH_SIZE // 2

                x_l, x_r = x - PATCH_SIZE // 2, x + PATCH_SIZE // 2 
                y_l, y_r = y - PATCH_SIZE // 2, y + PATCH_SIZE // 2 

                img = img_padded[y_l:y_r, x_l:x_r, :]

                if X:
                    background = np.copy(all_background_padded[y_l:y_r, x_l:x_r,:])

                    no_background_color = (255,0,255)
                    background_index = np.all(background != no_background_color, axis=-1)
                    selected_background = background[background_index]
                    background_l2 = np.expand_dims(255 - np.linalg.norm(background - img, axis=2) / np.sqrt(3.), axis=2)
                    background_mask = np.zeros((PATCH_SIZE, PATCH_SIZE,1), dtype=np.uint8)
                    background_mask[background_index] = 255
                    selected_background_l2 = background_l2[background_index]
  
                    if selected_background.size > 0:
                        selected_background_mean = np.mean(selected_background_l2)
                        selected_background_std  = np.std(selected_background_l2)
                    else:
                        selected_background_mean = np.mean(img)
                        selected_background_std  = np.std(img)                    

                    background_l2[~background_index] = \
                        np.random.normal(loc=selected_background_mean, scale=selected_background_std, size=(PATCH_SIZE**2-(selected_background.size//3),1))

                    img = np.concatenate([img, background_l2, background_mask], axis=2)

                if CO:
                    coarse = np.copy(all_coarse_padded_batch[idx-1, y_l:y_r, x_l:x_r])
                    img = np.concatenate([img, np.expand_dims(coarse, axis=2)], axis=2)

                if (input_size, input_size) != img.shape[:2]:
                    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                
                if img.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                    print(id)

                img = preprocess_for_model(img.astype(np.float32))
                flip = random.randint(0,1)
                if flip:
                    img = np.fliplr(img)

                img_batch[i,...] = copy.deepcopy(img)
                xy_batch.append(copy.deepcopy((x_l, x_r, y_l, y_r, flip, patch_id)))

                i += 1
                patch_id += 1

                if i == batch_size:

                    yield(img_batch)
                    #print("Yield:", car_id, idx)
                    i = 0

            car_xy_flip_batch.append(copy.deepcopy(xy_batch))

        #print("BYE!")
        # this is a workaround to make Keras max queue size happy 
        while True:
            yield(img_batch)
            #print("Yield:", car_id, idx)


    weighted_window = get_weighted_window(PATCH_SIZE)

    split_preffix = ''
    if args.test_total_splits != 1:
        split_preffix = str(args.test_current_split) + "_of_" + str(args.test_total_splits)

    with open(split_preffix + csv_filename, 'wb') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if args.test_current_split == 0:
            writer.writerow(['img', 'rle_mask'])

        all_coarse_padded = np.zeros((16,1280+PATCH_SIZE,1918+PATCH_SIZE), dtype=np.uint8)

        for car_id in tqdm(ids[args.test_current_split::args.test_total_splits]):

            car_xy_flip = []

            patches_probs = model.predict_generator(
                patches_generator(car_id, all_coarse_padded, car_xy_flip),
                steps = 16 * patches_per_image // batch_size,
                max_queue_size = 1)

            #print("FINISHED CALLING GEN")
            #print(patches_probs.shape)
            #print(car_xy_flip)
            #print(len(car_xy_flip))

            idx = 1
            for xy_batch in car_xy_flip:

                car_view_file = '{}_{:02d}'.format(car_id, idx) 

                probabilities_padded = np.zeros((1280+PATCH_SIZE,1918+PATCH_SIZE), dtype=np.float32)
                weights_padded       = np.zeros((1280+PATCH_SIZE,1918+PATCH_SIZE), dtype=np.float32)

                for (x_l, x_r, y_l, y_r, flip, patch_id) in xy_batch:
                    #print(x_l, x_r, y_l, y_r, flip)
                    patch_probs = np.squeeze(patches_probs[patch_id], axis=2)
                    if flip:
                        patch_probs = np.fliplr(patch_probs)    
                    probabilities_padded[y_l:y_r, x_l:x_r] += np.multiply(patch_probs, weighted_window)
                    weights_padded[y_l:y_r, x_l:x_r]       += weighted_window

                zero_weights = (weights_padded == 0)
                weights_padded[zero_weights] = 1.
                probabilities_padded /= weights_padded
                probabilities_padded[zero_weights] = all_coarse_padded[idx-1, zero_weights] / 255.

                probabilities = probabilities_padded[PATCH_SIZE//2:-PATCH_SIZE//2, PATCH_SIZE//2:-PATCH_SIZE//2]

                cv2.imwrite(join(save_pngs_to_folder, car_view_file + ".png"), probabilities*255.)

                rle = rle_encode(probabilities)
                writer.writerow([car_view_file + ".jpg", rle_to_string(rle)])
                idx += 1


initial_epoch = 0

if args.load_model:
    print("Loading model " + args.load_model)

    # monkey-patch loss so model loads ok
    # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
    import keras.losses
    import keras.metrics
    keras.losses.bce_dice_loss = bce_dice_loss
    keras.metrics.dice_loss = dice_loss
    keras.metrics.dice_loss100 = dice_loss100

    model = load_model(args.load_model, compile=False, custom_objects = { 'Scale' : Scale})
    match = re.search(r'patchesnet-([_a-zA-Z]+)-epoch(\d+)-.*', args.load_model)
    model_name = match.group(1).split("__")[0]
    initial_epoch = int(match.group(2)) + 1

    input_dimensions = model.get_input_shape_at(0)[1:]
    print(input_dimensions)
    assert input_dimensions[:2] == (args.input_size, args.input_size)

    name_dict = { 
        'rgb'    : (False, False, 3),
        'rgbCO'  : (False, True,  4),
        'rgbX'   : (True,  False, 5), 
        'rgbXCO' : (True,  True,  6) }

    X, CO, input_channels = name_dict[model.layers[1].name.split("_")[0]]

    if not args.test:
        assert args.use_background == X
        assert args.use_coarse == CO

else:
    model_name = args.model
    input_channels = 3
    if args.use_background:
        input_channels += 2
    if args.use_coarse:
        input_channels += 1

    model = getattr(u_net, 'get_'+ model_name)(input_shape=(input_size, input_size, input_channels))

if args.load_weights:
    model.load_weights(args.load_weights, by_name=True)

model.summary()

if args.suffix is None:
    suffix = "__rgb"
    if args.use_background:
        suffix += "X"
    if args.use_coarse:
        suffix += "CO"
else:
    suffix = "__" + args.suffix


if args.gpus != 1:
    model = to_multi_gpu(model,n_gpus=args.gpus)

if args.test:
    model_basename = args.load_model.split('/')[-1]
    mkdir_p('test_' + model_basename)
    test_model(model, ids, X, CO, 
        patches_per_image = args.test_patches_per_image, 
        batch_size = args.batch_size,
        csv_filename = model_basename + '.csv',
        save_pngs_to_folder = 'test_' + model_basename, 
        input_channels = input_channels)

else:

    callbacks = [ReduceLROnPlateau(monitor='val_dice_loss100',
                                     factor=0.5,
                                     patience=4,
                                     verbose=1,
                                     epsilon=1e-4,
                                     mode='max'),
                 ModelCheckpoint(monitor='val_dice_loss100',
                                 filepath=join(WEIGHTS_DIR,"patchesnet-"+ model_name + suffix + "-epoch{epoch:02d}-val_dice{val_dice_loss:.6f}"),
                                 save_best_only=False,
                                 save_weights_only=False,
                                 mode='max')]



    if args.optimizer == 'adam':
        optimizer=Adam(lr=args.learning_rate)
    elif args.optimizer == 'nadam':
        optimizer=Nadam(lr=args.learning_rate)
    elif args.optimizer == 'rmsprop':
        optimizer=RMSprop(lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer=SGD(lr=args.learning_rate, momentum=0.9)
    else:
        assert False

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_loss, dice_loss100])
    model.fit_generator(generator=generator(ids = ids_train_split, training=True),
                        steps_per_epoch=np.ceil(
                            float(len(ids_train_split)) / float(batch_size)) // args.fractional_epoch,
                        epochs=args.max_epoch,
                        initial_epoch = initial_epoch,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=generator(ids = ids_valid_split, training=False),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
