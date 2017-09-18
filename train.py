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
parser.add_argument('--max-epoch', type=int, default=200, help='Epoch to run')
parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch Size during training, e.g. -b 32')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='Initial learning rate, e.g. -l 1e-2')
parser.add_argument('-lw', '--load-weights', type=str, help='load model weights (and continue training)')
parser.add_argument('-lm', '--load-model', type=str, help='load model (and continue training)')
parser.add_argument('-c', '--cpu', action='store_true', help='force CPU usage')
parser.add_argument('-p', '--patch-size', type=int, default=256, help='Patch size, e.g -p 128')
parser.add_argument('-i', '--input-size', type=int, default=256, help='Network input size, e.g -i 256')
parser.add_argument('-ub', '--use-background', action='store_true', help='Use background as input to NN')
parser.add_argument('-o', '--optimizer', type=str, default='sgd', help='Optimizer to use: adam, nadam, sgd, e.g. -o adam')
parser.add_argument('-g', '--gpus', type=int, default=1, help='Use GPUs, e.g -g 2')
parser.add_argument('-at', '--augmentation-tps', action='store_true', help='TPS augmentation')
parser.add_argument('-af', '--augmentation-flips', action='store_true', help='Flips augmentation')
parser.add_argument('-s', '--suffix', type=str, default=None, help='Suffix for saving model name')
parser.add_argument('-m', '--model', type=str, default='dilated_unet', help='Use model, e.g. -m dilated_unet -m unet_256, unet_bg_256, largekernels')
parser.add_argument('-f', '--fractional-epoch', type=int, default=1, help='Reduce epoch steps by factor, e.g. -f 10 (after 10 epochs all samples would have been seen) ')

args = parser.parse_args()  

def preprocess_input_imagenet(img):
  return img.astype(np.float32) - np.float32([103.939, 116.779, 123.68])

preprocess_for_model = preprocess_input_imagenet if args.model == 'largekernels' else lambda x: x / 255.

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
        patches_dict = {rows[0]:(int(rows[1]),int(rows[2])) for rows in reader}
        background_dict = { }
        stats_dict      = { }

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

def randomHorizontalFlip(image, mask, background=None, u=0.5):
    if np.random.random() < u:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
        if background is not None:
          background = np.fliplr(background)

    if background is not None:
      return image, mask, background

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
                img = cv2.resize(
                    img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
              mask = cv2.imread(
                  join(TRAIN_FOLDER_MASKS, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)
              if (input_size,input_size) != mask.shape:
                mask = cv2.resize(
                    mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

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

                background = np.concatenate((background_l2, background_mask), axis=2)
              # img, mask = randomShiftScaleRotate(img, mask,
              #                                   shift_limit=(-0.0625, 0.0625),
              #                                   scale_limit=(-0.1, 0.1),
              #                                   rotate_limit=(-0, 0))
              if training:
                if not args.use_background and args.augmentation_tps:
                  #print(img.shape, mask.shape)
                  img, mask = tps({'img': img, 'mask': mask, 'seed': random.randint(0,1000)})
                  img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
                  mask = cv2.resize(mask, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
                  #if i == 0:
                  #  cv2.imwrite("img.jpg", img)
                  #i += 1
                  #print(img.shape, mask.shape)

                if args.augmentation_flips:
                  if args.use_background:
                    img, mask, background = randomHorizontalFlip(img, mask, background)
                  else:
                    img, mask = randomHorizontalFlip(img, mask)

              mask = np.expand_dims(mask, axis=2)
              if args.use_background:
                #cv2.imwrite("i.jpg", img)
                #cv2.imwrite("b.png", background[:,:,0])
                #cv2.imwrite("bm.png", background[:,:,1])

                x_batch.append(preprocess_for_model(np.concatenate((img, background), axis=2).astype(np.float32)))
              else:
                img = preprocess_for_model(img.astype(np.float32))
                x_batch.append(img)
              y_batch.append(mask)
              if img.shape != (PATCH_SIZE, PATCH_SIZE, 3):
                print(id)
          x_batch = np.array(x_batch, np.float32) 
          y_batch = np.array(y_batch, np.float32) / 255.
          yield x_batch, y_batch

initial_epoch = 0
if not args.load_model:
  model_name = args.model
  model = getattr(u_net, 'get_'+ model_name)(input_shape=(input_size, input_size, 5 if args.use_background else 3))
else:
  print("Loading model " + args.load_model)

  # monkey-patch loss so model loads ok
  # https://github.com/fchollet/keras/issues/5916#issuecomment-290344248
  import keras.losses
  import keras.metrics
  keras.losses.bce_dice_loss = bce_dice_loss
  keras.metrics.dice_loss = dice_loss
  keras.metrics.dice_loss100 = dice_loss100

  model = load_model(args.load_model, compile=False, custom_objects = { 'Scale' : Scale})
  match = re.search(r'patchesnet-([_a-z]+)-epoch(\d+)-.*', args.load_model)
  model_name = match.group(1).split("__")[0]
  initial_epoch = int(match.group(2)) + 1

if args.load_weights:
  model.load_weights(args.load_weights, by_name=True)

model.summary()
#model.get_layer('instance_normalization_1').name='instance_normalization_1_bg_axisNone'
#model.get_layer('conv2d_1').name='conv2d_1_bg'
#model.get_layer('conv2d_25').name='conv2d_25_bg'



#model.get_layer('conv2d_28').name='conv2d_28_deconv'
#model.get_layer('conv2d_22').name='conv2d_22_deconv'
#model.get_layer('conv2d_16').name='conv2d_16_deconv'
#model.get_layer('conv2d_13').name='conv2d_13_deconv'
#model.load_weights(filepath=join(WEIGHTS_DIR, 'patchesnet-unet_background_256-epoch00-val_dice0.994335'), by_name=True)
#model.load_weights(filepath=join(WEIGHTS_DIR, 'patchesnet-unet_256-epoch00-val_dice0.992748'), by_name=True)

if args.use_background and args.suffix is None:
  suffix = "bg"
  
suffix = "__" + args.suffix if args.suffix is not None else ""


callbacks = [ReduceLROnPlateau(monitor='val_dice_loss100',
                               factor=0.5,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4,
                               mode='max'),
             ModelCheckpoint(monitor='val_dice_loss100',
                             filepath=join(WEIGHTS_DIR,"patchesnet-"+ model_name + suffix + "-epoch{epoch:02d}-val_dice{val_dice_loss:.6f}"),
                             save_best_only=True,
                             save_weights_only=False,
                             mode='max')]

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
model.fit_generator(generator=generator(ids = ids_train_split, training=True),
                    steps_per_epoch=np.ceil(
                        float(len(ids_train_split)) / float(batch_size)) // args.fractional_epoch,
                    epochs=args.max_epoch,
                    initial_epoch = initial_epoch,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator(ids = ids_valid_split, training=False),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
