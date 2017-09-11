import cv2
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras.optimizers import SGD, Adam, RMSprop, Nadam

from sklearn.model_selection import train_test_split
import glob
import u_net
from u_net import bce_dice_loss, dice_loss, dice_loss100
import os
import scipy.misc as misc
import random
from os.path import join
import errno
import itertools
import argparse
from multi_gpu import to_multi_gpu

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
parser.add_argument('--max-epoch', type=int, default=50, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 32')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='Initial learning rate, e.g. -l 1e-2')
#parser.add_argument('-m', '--model', help='load hdf5 model (and continue training)')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-p', '--patch-size', type=int, default=256, help='Patch size, e.g -p 128')
parser.add_argument('-i', '--input-size', type=int, default=256, help='Network input size, e.g -i 256')
parser.add_argument('-ub', '--use-background', action='store_true', help='Use background as input to NN')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='Optimizer to use: adam, nadam, sgd, e.g. -o adam')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Use GPUs, e.g -g 2')
args = parser.parse_args()

if args.cpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

PATCH_SIZE = args.patch_size
TRAIN_FOLDER_PATCHES = join(ROOT_DIR, 'train_patches_' + str(PATCH_SIZE))
TRAIN_FOLDER_MASKS   = join(ROOT_DIR, 'train_patches_masks_' + str(PATCH_SIZE))
BACKGROUNDS_FOLDER   = join(ROOT_DIR, 'train_background_hq')
CSV_FILENAME         = join(ROOT_DIR, 'train_patches_' + str(PATCH_SIZE) + ".csv")

all_files = glob.glob(join(TRAIN_FOLDER_PATCHES, '*_*.jpg'))
ids = list(set([(x.split('/')[-1]).split('_')[0] for x in all_files]))
ids.sort()

if args.use_background:
   with open(CSV_FILENAME, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        patches_dict = {rows[0]:rows[1] for rows in reader}

ids_train_split, ids_valid_split = train_test_split(
    ids, test_size=0.1, random_state=13)

ids_train_split = [os.path.basename(x).split('.')[0]
                   for x in all_files if os.path.basename(x).split('_')[0] in ids_train_split]
ids_valid_split = [os.path.basename(x).split('.')[0]
                   for x in all_files if os.path.basename(x).split('_')[0] in ids_valid_split]

input_size = args.input_size
batch_size = args.batch_size 

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

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
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def train_generator():
    random.seed(13)
    while True:
        random.shuffle(ids_train_split)
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread(join(TRAIN_FOLDER_PATCHES, '{}.jpg'.format(id)))
                if (input_size, input_size, 3) != img.shape:
                  print(id)
                  img = cv2.resize(
                      img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                mask = cv2.imread(
                    join(TRAIN_FOLDER_MASKS, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
                if (input_size,input_size) != mask.shape:
                  print(id)
                  mask = cv2.resize(
                      mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                # img, mask = randomShiftScaleRotate(img, mask,
                #                                   shift_limit=(-0.0625, 0.0625),
                #                                   scale_limit=(-0.1, 0.1),
                #                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
                if img.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                  print(id)
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch:
                img = cv2.imread(join(TRAIN_FOLDER_PATCHES, '{}.jpg'.format(id)))
                if (input_size, input_size, 3) != img.shape:
                  print(id)
                  img = cv2.resize(
                      img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                mask = cv2.imread(
                    join(TRAIN_FOLDER_MASKS, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
                if (input_size,input_size) != mask.shape:
                  print(id)
                  mask = cv2.resize(
                      mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255.
            y_batch = np.array(y_batch, np.float32) / 255.
            yield x_batch, y_batch


callbacks = [EarlyStopping(monitor='val_dice_loss100',
                           patience=5,
                           verbose=1,
                           min_delta=1e-4,
                           mode='max'),
             ReduceLROnPlateau(monitor='val_dice_loss100',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4,
                               mode='max'),
             ModelCheckpoint(monitor='val_dice_loss100',
                             filepath=join(WEIGHTS_DIR,'patchesnet_unet256_noaug_sym_pad'),
                             save_best_only=True,
                             save_weights_only=True,
                             mode='max')]

model_name = 'unet_' + ('background_' if args.use_background else '') + str(PATCH_SIZE)
get_unet = getattr(u_net, 'get_'+ model_name)
model = get_unet(input_shape=(input_size, input_size, 3))
model.load_weights(filepath='weights/patchesnet_unet256_noaug_sym_pad', by_name=True)
model.summary()
if args.gpus != 1:
  model = to_multi_gpu(model,n_gpus=args.gpus)
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
model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(
                        float(len(ids_train_split)) / float(batch_size)),
                    epochs=args.max_epoch,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
