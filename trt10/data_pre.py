import numpy as np
import cv2

def get_image(impath, size):
    # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    # var = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
    var = np.array([0.5, 0.5, 0.5], dtype=np.float32)[:, None, None]
    iH, iW = size[0], size[1]
    im = cv2.imread(impath)
    # im = im[0:256,:]
    # im_cp = im.copy()
    img = im[:, :, ::-1]
    # orgH, orgW, _ = img.shape
    img = cv2.resize(img, (iW, iH)).astype(np.float32)
    img = img.transpose(2, 0, 1)
    # img = img.transpose(2, 0, 1) / 255.
    img = (img - mean) / var
    return img
