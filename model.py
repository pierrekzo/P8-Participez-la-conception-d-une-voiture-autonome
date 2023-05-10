import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence, img_to_array
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models as sm

def predict(img_path):

    dice_loss = sm.losses.DiceLoss()
    mean_iou = sm.metrics.IOUScore()
    mean_dice = sm.metrics.FScore(beta=1)
    model = load_model('/content/drive/MyDrive/Projets OCR/P8-IA/Models/UNET', custom_objects= {'iou_score': mean_iou, 'f1-score':mean_dice, 'dice_loss':dice_loss})

    img = img_to_array(load_img(img_path, target_size=(256,256,3)))/255
    img = np.expand_dims(img,axis=0)
    original_img = img_to_array(load_img(img_path))/255.


    # Predict
    pred_mask = model.predict(img)
    pred_mask = np.argmax(pred_mask, axis=-1)
    pred_mask = np.expand_dims(pred_mask, axis=-1)
    pred_mask = np.squeeze(pred_mask)
    # Use interpolation inter_nearest to use integer with cv2
    pred_mask = cv2.resize(pred_mask, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)
    fig = plt.figure(figsize=(20, 20))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('orginal image')
    ax1.imshow(original_img)
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('predicted_Mask')
    ax2.imshow(pred_mask)