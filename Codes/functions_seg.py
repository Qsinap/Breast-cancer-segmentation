import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import math
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import scipy.io as sio
import pandas as pd
import time
import MultiResUNet as multi
import cv2
from keras_unet.models import custom_unet
from segmentation_models import Unet, Linknet, FPN, PSPNet

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

def tversky(y_true, y_pred,  smooth=1.):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def convert_to_logits(y_pred):
    y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    return tf.math.log(y_pred / (1 - y_pred))

def weighted_cross_entropyloss(y_true, y_pred, beta = 0.25):
    y_pred = convert_to_logits(y_pred)
    pos_weight = beta / (1 - beta)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
    return tf.reduce_mean(loss)

def log_cosh_dice_loss(y_true, y_pred):
    x = dice_coef_loss(y_true, y_pred)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)

def sensitivity(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())

def false_positive_rate(y_true, y_pred):
    return 1-specificity(y_true, y_pred)

def contour(ima1, mas1):
  bina=((mas1!=0)*255).astype('uint8')
  cont,_=cv2.findContours(bina, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  rgb=cv2.cvtColor((ima1*255).astype('uint8'), cv2.COLOR_GRAY2BGR)
  dra=cv2.drawContours(rgb,cont, -1, (255,0,0),2)
  return dra

def plot_mamo(ima1,mas1):
  plt.figure(figsize=(15,7))
  plt.subplot(1,3,1)
  plt.imshow(ima1, cmap='gray')
  plt.axis('off')

  plt.subplot(1,3,2)
  plt.imshow(contour(ima1, mas1))
  plt.axis('off')

  plt.subplot(1,3,3)
  plt.imshow(mas1,cmap='hot')
  plt.axis('off')

def validation_modela_and_save(csv_n, ix, jx, mat_n):
  y_hat=model.predict(x_test)

  acc=np.mean((y_test>0.5)==(y_hat>0.5))
  jxd=np.mean(jaccard_distance(y_test, y_hat))
  dic=np.mean(dice_coef(y_test, y_hat))
  sensi=float(sensitivity(y_test, y_hat))
  speci=float(specificity(y_test, y_hat))
  total_p=model.count_params()


  #data frame
  df2=pd.read_csv(csv_n)
  df2=df2.append({'tumor_type': tumor_t,
                'run_n': ix,
                'network': network,
                'optimizer': optimizer,
                'loss': jx,
                'epochs': epochs,
                'total_parameters': total_p,
                'time': toc,'augm': augmentation,
                'jaccard_distances': jxd,
                'acc': acc,
                'sensitivity': sensi,
                'specificity': speci,
                'FPR': 1-speci,
                'dice_coef': dic,
                'result_mat': mat_n} , ignore_index=True)
  df2=df2.drop(df2.columns[:np.where(df2.columns=='tumor_type')[0][0]], axis=1) 
  df2.to_csv(csv_n)

def read_model(net, back, inputshape=(256,256,3)):
  not_p=True
  if net=='multiresunet':
    model=multi.MultiResUnet(256,256,3)
    not_p=False
  
  if net=='link':
    model=Linknet(back, inputshape)
    not_p=False
  
  if net=='unet':
    model=Unet(back, inputshape)
    not_p=False

  if net=='Base_unet':
    model=custom_unet(inputshape)
    not_p=False

  if not_p:
    print('Network name does not exist')

  return model


