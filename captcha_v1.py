import os
from glob import glob
from pathlib import Path

from sklearn.preprocessing import LabelEncoder
import numpy as np

from torch import nn, softmax,argmax,load
from torch.utils.data import DataLoader

import config
import dataset
from model import CaptchaModel

lbl_enc = LabelEncoder()
lbl_enc.classes_ = np.load('./checkpoints/captcha_v1/classes.npy')

def evaluate():
    files =[]
    for img_ext in config.ALLOWED_EXTENSIONS :
        files.extend(glob(os.path.join(config.IMAGES_UPLOADED_PATH,"*.{}".format(img_ext))))
    files.sort(key=os.path.getctime,reverse=True)
    test_img = files[:1]
        
    test_dataset = dataset.ClassificationDataset(image_paths=test_img,resize=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
    test_loader = DataLoader(test_dataset,batch_size=1,num_workers=0,)

    model = CaptchaModel(num_chars=len(lbl_enc.classes_))
    model.to(config.DEVICE)
    
        
    model.load_state_dict(load('./checkpoints/captcha_v1/captcha_v1.pth',map_location= config.DEVICE))    
    model.eval()

    for data in test_loader:
        data["images"] = data["images"].to(config.DEVICE)
        prediction, _ = model(data["images"])
        prediction_output = decode_predictions(prediction,lbl_enc)
        return prediction_output


# UTILS FUNCTIONS 
def remove_duplicates(prediction):
    if len(prediction)<2:
        return prediction

    prediction_list = [ch for ch in prediction]
    output=[]
    for i in range (len(prediction_list)-1):
        temp = prediction_list[i]
        if prediction_list[i+1] != temp :
            output.append(temp)
        else :
            continue
        
    if prediction_list[-1] != config.PATTERN:
        output.append(prediction_list[-1])
        
    return ''.join([x for x in output if x!=config.PATTERN]) 

def decode_predictions(preds, encoder):
    preds = preds.permute(1, 0, 2)
    preds = softmax(preds, 2)
    preds = argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []
    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k - 1
            if k == -1:
                temp.append(config.PATTERN)
            else:
                p = encoder.inverse_transform([k])[0]
                temp.append(p)
        tp = "".join(temp)
        cap_preds.append(remove_duplicates(tp))
    return cap_preds