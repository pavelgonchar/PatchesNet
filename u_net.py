from __future__ import division, print_function
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop, Nadam
from keras.losses import binary_crossentropy
import keras.backend as K
from multi_gpu import to_multi_gpu
from keras.layers.merge import add
from keras_contrib.layers.normalization import InstanceNormalization
from keras.applications import ResNet50

from keras.layers import Input, Conv2D, Conv2DTranspose
from keras.layers import MaxPooling2D, Cropping2D, Concatenate
from keras.layers import Lambda, Activation, BatchNormalization, Dropout
from keras.models import Model
from keras import initializers
from keras.engine import Layer, InputSpec
import math


def weighted_bce_loss(y_true, y_pred, weight):
    # avoiding overflow
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    logit_y_pred = K.log(y_pred / (1. - y_pred))

    # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
    loss = (1. - y_true) * logit_y_pred + (1. + (weight - 1.) * y_true) * \
        (K.log(1. + K.exp(-K.abs(logit_y_pred)))
         + K.maximum(-logit_y_pred, 0.))
    return K.sum(loss) / K.sum(weight)


def weighted_dice_loss(y_true, y_pred, weight):
    smooth = 1.
    w, m1, m2 = weight * weight, y_true, y_pred
    intersection = (m1 * m2)
    score = (2. * K.sum(w * intersection) + smooth) / \
        (K.sum(w * m1) + K.sum(w * m2) + smooth)
    loss = 1. - K.sum(score)
    return loss


def weighted_bce_dice_loss(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.cast(y_pred, 'float32')
    # if we want to get same size of output, kernel size must be odd number
    averaged_mask = K.pool2d(
        y_true, pool_size=(11, 11), strides=(1, 1), padding='same', pool_mode='avg')
    border = K.cast(K.greater(averaged_mask, 0.005), 'float32') * \
        K.cast(K.less(averaged_mask, 0.995), 'float32')
    weight = K.ones_like(averaged_mask)
    w0 = K.sum(weight)
    weight += border * 2
    w1 = K.sum(weight)
    weight *= (w0 / w1)
    # loss = weighted_bce_loss(y_true, y_pred, weight) + \
    loss = weighted_dice_loss(y_true, y_pred, weight)
    return loss


def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_loss100(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (200. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def jaccard_coef(y_true, y_pred):
    # __author__ = Vladimir Iglovikov
    smooth = 1e-12
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def bce_dice_loss(y_true, y_pred):
    # return (binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true,
    # y_pred))) /2
    return 1 - dice_loss(y_true, y_pred)
    # return 1 -jaccard_coef(y_true, y_pred)


def get_unet_128(input_shape=(128, 128, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=SGD(lr=0.01, momentum=0.9),
                  loss=bce_dice_loss, metrics=[dice_loss, dice_loss100])

    return model


def get_unet_256(input_shape=(256, 256, 3), naive_upsampling= True ):

    inputs = Input(shape=input_shape)
    bg_preffix_dict = { 
        3 : 'rgb_',     # RGB 
        4 : 'rgbCO_',   # RGB + coarse mask
        5 : 'rgbX_',    # RGB + BG 
        6 : 'rgbXCO_' } # RGB + BG + coarse mask
    bg_preffix = bg_preffix_dict[input_shape[2]]

    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    inputs_normalized = InstanceNormalization(axis=bn_axis, name=bg_preffix + 'instancenormalization1')(inputs)

    # 256

    # receptive 365
    down0 = Conv2D(32, (3, 3), padding='same', name=bg_preffix + "conv1")(inputs_normalized)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    # receptive 155
    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    # receptive 75
    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

    # receptive 35
    # 32
    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    # receptive 15
    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    # receptive 5
    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    if naive_upsampling:
        up4 = UpSampling2D((2, 2))(center)
    else:
        up4 = Conv2DTranspose(1024, kernel_size=(2,2), strides=(2,2))(center)  
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    if naive_upsampling:
        up3 = UpSampling2D((2, 2))(up4)
    else:
        up3 = Conv2DTranspose(512, kernel_size=(2,2), strides=(2,2))(up4)        
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    if naive_upsampling:
        up2 = UpSampling2D((2, 2))(up3)
    else:
        up2 = Conv2DTranspose(256, kernel_size=(2,2), strides=(2,2))(up3)    
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64
    if naive_upsampling:
        up1 = UpSampling2D((2, 2))(up2)
    else:
        up1 = Conv2DTranspose(128, kernel_size=(2,2), strides=(2,2))(up2)

    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    down0xtra = Conv2D(32, (3, 3), padding='same', name=bg_preffix + "down0_xtra_conv1")(inputs_normalized)
    down0xtra = BatchNormalization()(down0xtra)
    down0xtra = Activation('relu')(down0xtra)
    down0xtra = Conv2D(32, (3, 3), padding='same')(down0xtra)
    down0xtra = BatchNormalization()(down0xtra)
    down0xtra = Activation('relu')(down0xtra)
    down0xtra = Conv2D(32, (3, 3), padding='same')(down0xtra)
    down0xtra = BatchNormalization()(down0xtra)
    down0xtra = Activation('relu')(down0xtra)

    if naive_upsampling:
        up0 = UpSampling2D((2, 2))(up1)
    else:
        up0 = Conv2DTranspose(64, kernel_size=(2,2), strides=(2,2))(up1)

    up0 = concatenate([down0, up0, down0xtra], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    return model

def get_resunet(input_shape=(256, 256, 3), naive_upsampling= True, full_residual = False):

    inputs = Input(shape=input_shape)
    bg_preffix_dict = { 
        3 : 'rgb_',     # RGB 
        4 : 'rgbCO_',   # RGB + coarse mask
        5 : 'rgbX_',    # RGB + BG 
        6 : 'rgbXCO_' } # RGB + BG + coarse mask
    bg_preffix = bg_preffix_dict[input_shape[2]]

    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    inputs_normalized = InstanceNormalization(axis=bn_axis, name=bg_preffix + 'instancenormalization1')(inputs)

    down = inputs_normalized

    down_blocks = [32, 64, 128,  256, 512, 1024]
    skips = []

    scale = False

    #  7   -> 14 + 7 -> 42 + 7 -> 98 +7 -> 210 + 7 = 217
    # 512  -> 256    -> 128    -> 64    -> 32

    trf = 0
    erf = 0
    residual_filter_factor = 4 if full_residual else 1
    for i, block in enumerate(down_blocks):
        first = i == 0
        last  = i == (len(down_blocks) -1)

        dilation = [1,1,1]
        k = 3

        def unet_block(input, kernel_size, filters, stage, block, filter_start=0, preffix=""):
            x = input
            for i, nb_filter in enumerate(filters):
                suffix = str(stage) + "_" + block + "_" + str(i+filter_start)
                x = Conv2D(nb_filter, kernel_size, padding='same', name=preffix + 'conv_' + suffix)(x)
                x = BatchNormalization(name=preffix + 'bn_' + suffix)(x)
                x = Activation('relu', name=preffix + 'act_' + suffix)(x)
            return x

        if not full_residual:
            down = unet_block(down, k, [block, block], stage=i, block='down', preffix = bg_preffix if first else '')
            down = conv_block(down, k, [block, block, block*residual_filter_factor], stage=i, block='down', strides=(1, 1), preffix=bg_preffix if first else '', zero_padding=False, scale=scale)
        
        if full_residual:
            down = conv_block(down, k, [block, block, block*residual_filter_factor], stage=i, block='down', strides=(1, 1), preffix=bg_preffix if first else '', zero_padding=False, scale=scale)
            down = identity_block(down, k, [block, block, block*residual_filter_factor], stage=i, block='down0', preffix='', zero_padding=False, scale=scale)
            down = identity_block(down, k, [block, block, block*residual_filter_factor], stage=i, block='down1', preffix='', zero_padding=False, scale=scale)
        rf = 1 + (k-1) * dilation[0] + (k-1) * dilation[1]+ (k-1) * dilation[2]
        trf += rf
        erf += rf / math.sqrt(3)

        if not last:
            skips.append(down)
            down = MaxPooling2D((2, 2), strides=(2, 2))(down)
            trf *= 2
            erf *= 2

    # see http://www.cs.toronto.edu/~wenjie/papers/nips16/top.pdf
    print("Theoretical receptive field: " + str(trf) + " pixels")
    print("Effective   receptive field: " + str(erf) + " pixels")

    up = down
    for i, block in enumerate(down_blocks[:-1][::-1]):
        if naive_upsampling:
            up = UpSampling2D((2, 2))(up)
        else:
            up = Conv2DTranspose(block, kernel_size=(2,2), strides=(2,2))(up)  
        up = concatenate([up, skips.pop()], axis=3)
        if not full_residual:
            up = Conv2D(block, 3, padding='same')(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
            up = Conv2D(block, 3, padding='same')(up)
            up = BatchNormalization()(up)
            up = Activation('relu')(up)
        up = conv_block(up, 3, [block//1, block//1, block*residual_filter_factor], stage=i, block='up', strides=(1, 1), preffix='', zero_padding=False, scale=scale)
        if full_residual:
            up = identity_block(up, 3, [block, block, block*residual_filter_factor], stage=i, block='up0', preffix='', zero_padding=False, scale=scale)
            up = identity_block(up, 3, [block, block, block*residual_filter_factor], stage=i, block='up1', preffix='', zero_padding=False, scale=scale)
    
    kernel_sigmoid = 7
    classify = Conv2D(1, (kernel_sigmoid, kernel_sigmoid), padding='same', activation='sigmoid', name='conv_sigmoid_' + str(kernel_sigmoid))(up)

    model = Model(inputs=inputs, outputs=classify)

    return model

def get_unet_background_256(input_shape=(256, 256, 6),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    inputs_normalized = InstanceNormalization(axis=3)(inputs)
    # 256

    down0 = Conv2D(32, (3, 3), padding='same')(inputs_normalized)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    down0xtra = Conv2D(32, (3, 3), padding='same')(inputs_normalized)
    down0xtra = BatchNormalization()(down0xtra)
    down0xtra = Activation('relu')(down0xtra)
    down0xtra = Conv2D(32, (3, 3), padding='same')(down0xtra)
    down0xtra = BatchNormalization()(down0xtra)
    down0xtra = Activation('relu')(down0xtra)
    down0xtra = Conv2D(32, (3, 3), padding='same')(down0xtra)
    down0xtra = BatchNormalization()(down0xtra)
    down0xtra = Activation('relu')(down0xtra)

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0, down0xtra], axis=3)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    # model = to_multi_gpu(model,n_gpus=8)

    return model

def get_unet_512(input_shape=(512, 512, 3),
                 num_classes=1):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16 * 2, (7, 7), padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(16 * 2, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32 * 2, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(32 * 2, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64 * 2, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(64 * 2, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128 * 2, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(128 * 2, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256 * 2, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(256 * 2, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512 * 2, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(512 * 2, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024 * 2, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(1024 * 2, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512 * 2, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512 * 2, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(512 * 2, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256 * 2, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256 * 2, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(256 * 2, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128 * 2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128 * 2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(128 * 2, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64 * 2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64 * 2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(64 * 2, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32 * 2, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32 * 2, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(32 * 2, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16 * 2, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16 * 2, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(16 * 2, (7, 7), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    # model = to_multi_gpu(model,n_gpus=8)

    model.compile(optimizer=SGD(lr=0.001, momentum=0.99),
                  loss=bce_dice_loss, metrics=[dice_loss])

    return model


def get_unet_1024(input_shape=(1024, 1024, 3),
                  num_classes=1, mult=2):
    inputs = Input(shape=input_shape)
    # 1024

    # preprocess = BatchNormalization(center=False, scale=False,
    # name='preprocess')(inputs)

    # receptive 1275
    down0b = Conv2D(4 * mult, (3, 3), padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)
    down0b = Conv2D(4 * mult, (3, 3), padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = Activation('relu')(down0b)

    # receptive 1270
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    # receptive 635
    down0a = Conv2D(8 * mult, (3, 3), padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    down0a = Conv2D(8 * mult, (3, 3), padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = Activation('relu')(down0a)
    # receptive 630
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    # receptive 315
    down0 = Conv2D(16 * mult, (3, 3), padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)
    down0 = Conv2D(16 * mult, (3, 3), padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = Activation('relu')(down0)

    # receptive 310
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    # receptive 155
    down1 = Conv2D(32 * mult, (3, 3), padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    down1 = Conv2D(32 * mult, (3, 3), padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = Activation('relu')(down1)
    # receptive 150
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    # receptive 75
    down2 = Conv2D(64 * mult, (3, 3), padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    down2 = Conv2D(64 * mult, (3, 3), padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = Activation('relu')(down2)
    # receptive 70
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    # receptive 35
    down3 = Conv2D(128 * mult, (3, 3), padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    down3 = Conv2D(128 * mult, (3, 3), padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = Activation('relu')(down3)
    # receptive 30
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    # receptive 15
    down4 = Conv2D(256 * mult, (3, 3), padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)
    down4 = Conv2D(256 * mult, (3, 3), padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = Activation('relu')(down4)

    # receptive 10
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    # receptive 5
    center = Conv2D(512 * mult, (3, 3), padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    center = Conv2D(512 * mult, (3, 3), padding='same')(center)
    center = BatchNormalization()(center)
    center = Activation('relu')(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(256 * mult, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256 * mult, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    up4 = Conv2D(256 * mult, (3, 3), padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = Activation('relu')(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(128 * mult, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128 * mult, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    up3 = Conv2D(128 * mult, (3, 3), padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = Activation('relu')(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(64 * mult, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64 * mult, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    up2 = Conv2D(64 * mult, (3, 3), padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = Activation('relu')(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(32 * mult, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32 * mult, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    up1 = Conv2D(32 * mult, (3, 3), padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = Activation('relu')(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(16 * mult, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(16 * mult, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    up0 = Conv2D(16 * mult, (3, 3), padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = Activation('relu')(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(8 * mult, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(8 * mult, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    up0a = Conv2D(8 * mult, (3, 3), padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = Activation('relu')(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(4 * mult, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(4 * mult, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    up0b = Conv2D(4 * mult, (3, 3), padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = Activation('relu')(up0b)
    # 1024

    # res_up0b = up0b
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res1_1')(up0b)
    # up0b = BatchNormalization(name='res1_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res1_2')(up0b)
    # up0b = add([res_up0b, up0b])

    # res_up0b = up0b
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res2_1')(up0b)
    # up0b = BatchNormalization(name='res2_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res2_2')(up0b)
    # up0b = add([res_up0b, up0b])

    # res_up0b = up0b
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res3_1')(up0b)
    # up0b = BatchNormalization(name='res3_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res3_2')(up0b)
    # up0b = add([res_up0b, up0b])

    # res_up0b = up0b
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res4_1')(up0b)
    # up0b = BatchNormalization(name='res4_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res4_2')(up0b)
    # up0b = add([res_up0b, up0b])

    # res_up0b = up0b
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res5_1')(up0b)
    # up0b = BatchNormalization(name='res5_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res5_2')(up0b)
    # up0b = add([res_up0b, up0b])

    # res_up0b = up0b
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res6_1')(up0b)
    # up0b = BatchNormalization(name='res6_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult, (3, 3), padding='same', name='res6_2')(up0b)
    # up0b = add([res_up0b, up0b])

    up0b = concatenate([up0b, inputs])
    # res_up0b = up0b
    # up0b = Conv2D(4*mult+3, (3, 3), padding='same', name='res6_1')(up0b)
    # up0b = BatchNormalization(name='res6_1_bn')(up0b)
    # up0b = Activation('relu')(up0b)
    # up0b = Conv2D(4*mult+3, (3, 3), padding='same', name='res6_2')(up0b)
    # up0b = add([res_up0b, up0b])

    classify = Conv2D(
        num_classes, (1, 1), activation='sigmoid', name='newsigmoid')(up0b)
    # classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    # model = to_multi_gpu(model,n_gpus=2)

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9),
                  loss=bce_dice_loss, metrics=[dice_loss, dice_loss100])
    # model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss=bce_dice_loss,
    # metrics=[dice_loss2])

    return model

def downsampling_block(input_tensor, filters, padding='valid',
                       batchnorm=False, dropout=0.0):
    _, height, width, _ = K.int_shape(input_tensor)
    print(height, width)
    #assert height % 2 == 0
    #assert width % 2 == 0

    x = Conv2D(filters, kernel_size=(3,3), padding=padding,
               dilation_rate=1)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding, dilation_rate=2)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return MaxPooling2D(pool_size=(2,2))(x), x

def upsampling_block(input_tensor, skip_tensor, filters, padding='valid',
                     batchnorm=False, dropout=0.0):
    #x = UpSampling2D((2, 2))(input_tensor)
    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)


    # compute amount of cropping needed for skip_tensor
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)

    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    print(K.int_shape(x))
    print(K.int_shape(y))

    x = Concatenate(axis=3)([x, y])

    # no dilation in upsampling convolutions
    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x

def get_dilated_unet(input_shape=(256, 256, 3), features=64, depth=3,
                 temperature=1.0, padding='same', batchnorm=True,
                 dropout=0.0, dilation_layers=3):
    """Generate `dilated U-Net' model where the convolutions in the encoding and
    bottleneck are replaced by dilated convolutions. The second convolution in
    pair at a given scale in the encoder is dilated by 2. The number of
    dilation layers in the innermost bottleneck is controlled by the
    `dilation_layers' parameter -- this is the `context module' proposed by Yu,
    Koltun 2016 in "Multi-scale Context Aggregation by Dilated Convolutions"

    Arbitrary number of input channels and output classes are supported.

    Arguments:
      height  - input image height (pixels)
      width   - input image width  (pixels)
      channels - input image features (1 for grayscale, 3 for RGB)
      classes - number of output classes (2 in paper)
      features - number of output features for first convolution (64 in paper)
          Number of features double after each down sampling block
      depth  - number of downsampling operations (4 in paper)
      padding - 'valid' (used in paper) or 'same'
      batchnorm - include batch normalization layers before activations
      dropout - fraction of units to dropout, 0 to keep all units
      dilation_layers - number of dilated convolutions in innermost bottleneck

    Output:
      Dilated U-Net model expecting input shape (height, width, maps) and
      generates output with shape (output_height, output_width, classes).
      If padding is 'same', then output_height = height and
      output_width = width.

    """
    height, width, channels = input_shape

    input = Input(shape=input_shape)
    x = InstanceNormalization(axis=3)(input)

    inputs = input

    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding,
                                   batchnorm, dropout)
        skips.append(x0)
        features *= 2

    dilation_rate = 1
    for n in range(dilation_layers):
        x = Conv2D(filters=features, kernel_size=(3,3), padding=padding,
                   dilation_rate=dilation_rate)(x)
        x = BatchNormalization()(x) if batchnorm else x
        x = Activation('relu')(x)
        x = Dropout(dropout)(x) if dropout > 0 else x
        dilation_rate *= 2

    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skips[i], features, padding,
                             batchnorm, dropout)

    segmentation = Conv2D(filters=1, activation='sigmoid', kernel_size=(1,1))(x)


    return Model(inputs=inputs, outputs=segmentation)

def gcn(i, k, n, filters=1):
    left  = Conv2D(filters, kernel_size=(k,1), padding='same', name=n + '_l0')(i)
    left  = Conv2D(filters, kernel_size=(1,k), padding='same', name=n + '_l1')(left)
    right = Conv2D(filters, kernel_size=(1,k), padding='same', name=n + '_r0')(i)
    right = Conv2D(filters, kernel_size=(k,1), padding='same', name=n + '_r1')(right)
    return add([left, right])

def br(i, n, filters=1, activation='relu'):
    res = Conv2D(filters, kernel_size=(3,3), padding='same', activation=activation, name=n + '_c0')(i)
    res = Conv2D(filters, kernel_size=(3,3), padding='same', name=n + '_c1')(res)
    res = Conv2D(i._keras_shape[-1], kernel_size=(1,1), padding='same', name=n + '_c2')(res)
    return add([res, i])

class Scale(Layer):
    '''Custom Layer for ResNet used for BatchNormalization.
    
    Learns a set of weights and biases used for scaling the input data.
    the output consists simply in an element-wise multiplication of the input
    and a sum of a set of constants:

        out = in * gamma + beta,

    where 'gamma' and 'beta' are the weights and biases larned.

    # Arguments
        axis: integer, axis along which to normalize in mode 0. For instance,
            if your input tensor has shape (samples, channels, rows, cols),
            set axis to 1 to normalize per feature map (channels axis).
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
        beta_init: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_init: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
    '''
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def identity_block(input_tensor, kernel_size, filters, stage, block, preffix='', activation='relu', zero_padding=False, scale=True):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = preffix + 'res' + str(stage) + block + '_branch'
    bn_name_base = preffix + 'bn' + str(stage) + block + '_branch'
    scale_name_base = preffix + 'scale' + str(stage) + block + '_branch'

    padding = 'valid' if zero_padding else 'same'

    x = Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', use_bias=False, padding=padding)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    if scale: x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation(activation, name=conv_name_base + '2a_relu')(x)

    if zero_padding:
        x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False, padding=padding)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    if scale: x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation(activation, name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False, padding=padding)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    if scale: x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = add([x, input_tensor], name=preffix + 'res' + str(stage) + block)
    x = Activation(activation, name=preffix + 'res' + str(stage) + block + '_relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), preffix='', activation='relu', zero_padding=True, scale=True):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    '''
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = preffix + 'res' + str(stage) + block + '_branch'
    bn_name_base = preffix + 'bn' + str(stage) + block + '_branch'
    scale_name_base = preffix + 'scale' + str(stage) + block + '_branch'

    padding = 'valid' if zero_padding else 'same'

    x = Conv2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False, padding=padding)(input_tensor)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    if scale: x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = Activation(activation, name=conv_name_base + '2a_relu')(x)

    if zero_padding:
        x = ZeroPadding2D((1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False, padding=padding)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    if scale: x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = Activation(activation, name=conv_name_base + '2b_relu')(x)

    x = Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', use_bias=False, padding=padding)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    if scale: x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=False, padding=padding)(input_tensor)
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    if scale: shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = add([x, shortcut], name=preffix + 'res' + str(stage) + block)
    x = Activation(activation, name=preffix + 'res' + str(stage) + block + '_relu')(x)
    return x

def get_largekernels(input_shape=(256, 256, 3), k=15):

    '''Instantiate the ResNet152 architecture,
    # Arguments
        weights_path: path to pretrained weight file
    # Returns
        A Keras model instance.
    '''
    eps = 1.1e-5

    # Handle Dimension Ordering for different backends
    global bn_axis
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
        img_input = Input(shape=input_shape, name='data')
    else:
        bn_axis = 1
        img_input = Input(shape=input_shape, name='data')

    res1 = Scale(axis=bn_axis, name='lkm_scale_conv1')(img_input)
    act = 'relu'
    res1 = conv_block(res1, 3, [32, 32, 128], stage=1, block='a', strides=(1, 1), preffix='lkm_', activation=act)
    res1 = identity_block(res1, 3, [32, 32, 128], stage=1, block='b', preffix='lkm_', activation=act)
    res1 = identity_block(res1, 3, [32, 32, 128], stage=1, block='c', preffix='lkm_', activation=act)
            
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = Activation(act, name='conv1_relu')(x)
    res2 = x

    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1', padding='same')(x)
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), activation=act)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b', activation=act)
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c', activation=act)

    res3 = x

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', activation=act)
    for i in range(1,8):
        x = identity_block(x, 3, [128, 128, 512], stage=3, block='b'+str(i), activation=act)

    res4 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', activation=act)
    for i in range(1,36):
        x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b'+str(i), activation=act)

    res5 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', activation=act)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', activation=act)
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', activation=act)

    res6 = x

    ff = 64
    act = 'elu'


    br6  = br(gcn(res6, k, n='lkm_gcn6_', filters = ff//(2**5)), filters = ff//(2**5), n='lkm_br6_', activation=act) # 2048 -> 1024
    br5  = br(gcn(res5, k, n='lkm_gcn5_', filters = ff//(2**5)), filters = ff//(2**5), n='lkm_br5_', activation=act)
    br4  = br(gcn(res4, k, n='lkm_gcn4_', filters = ff//(2**4)), filters = ff//(2**4), n='lkm_br4_', activation=act)
    br3  = br(gcn(res3, k, n='lkm_gcn3_', filters = ff//(2**3)), filters = ff//(2**3), n='lkm_br3_', activation=act)
    br2  = br(gcn(res2, k, n='lkm_gcn2_', filters = ff//(2**2)), filters = ff//(2**2), n='lkm_br2_', activation=act)
    br1  = br(gcn(res1, k, n='lkm_gcn1_', filters = ff//(2**1)), filters = ff//(2**1), n='lkm_br1_', activation=act)

    u6   = Conv2DTranspose( ff//(2**5), kernel_size=(2,2), strides=(2,2), name='lkm_u6_')(br6)  #  32 x  32
    br5a = br(add([u6,br5]), n='lkm_br5a_', filters=ff//(2**5), activation=act)
    u5   = Conv2DTranspose( ff//(2**4), kernel_size=(2,2), strides=(2,2), name='lkm_u5_')(br5a)  #  32 x  32
    br4a = br(add([u5,br4]), n='lkm_br4a_', filters=ff//(2**4), activation=act)
    u4  =  Conv2DTranspose( ff//(2**3), kernel_size=(2,2), strides=(2,2), name='lkm_u4_')(br4a) #  64 x  64
    br3a = br(add([u4,br3]), n='lkm_br3a_', filters=ff//(2**3), activation=act)
    u3  =  Conv2DTranspose( ff//(2**2), kernel_size=(2,2), strides=(2,2), name='lkm_u3_')(br3a) # 128 x 128
    br2a = br(add([u3,br2]), n='lkm_br2a_', filters=ff//(2**2), activation=act)
    u2  =  Conv2DTranspose( ff//(2**1), kernel_size=(2,2), strides=(2,2), name='lkm_u2_')(br2a) # 256 x 256
    br1a = br(add([u2,br1]), n='lkm_br1a_', filters=ff//(2**1), activation=act)

    segmentation = Conv2D(1, (1, 1), activation='sigmoid', name='lkm_segmentation_')(br1a)

    model = Model(inputs=img_input, outputs=segmentation)
    #model.load_weights("resnet152_weights_tf.h5", by_name=True)

    for layer in model.layers:
        if layer.name.split("_")[0] in ['lkm', 'bn5a','bn5b', 'bn5c', 'res5a','res5b', 'res5c', 'scale5a','scale5b','scale5c'] :
            layer.trainable = True
        else:
            layer.trainable = False


    return model
