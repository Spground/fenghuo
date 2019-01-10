from multiprocessing import Process, JoinableQueue
import os, time, random
import argparse
import os
import sys
import shutil
import json
import glob
import signal
import pickle
import csv
import torch
import torch.nn as nn
from data_loader import VideoFolder
from callbacks import PlotLearning, MonitorLRDecay, AverageMeter
from model import ConvColumn
from c3d_model import C3D
from c3d_model_deep import C3DDeep
from c3d_model_deep_deep import *
from torchvision.transforms import *
import numpy as np

from time import time
import time as tm
import cv2
from PIL import Image, ImageDraw, ImageFont

from utils import detector_utils as detector_utils
import tensorflow as tf
import datetime
 
config_file =  "configs/single_example_test_config.json"   
device = torch.device("cpu")
label = ["向左划","向右划","向下划","向上划","手推向远处","手从远处拉回","两根手指向左滑动","两根手指向右滑动","两根手指向下滑动",
"两根手指向上滑动","两根手指推向远处","两根手指从远处拉回","手向前滚动","手向后滚动","顺时针转手","逆时针转手","用手放大","用手缩小",
"两根手指放大","两根手指缩小","拇指向上","拇指向下","摇动手掌","停止","抖动手指","没有手势","其他手势"]

font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
txt_location = (440,60)
fontScale = 1
fontColor = (255,0,0)
lineType = 2
diss_time = 75 # about 2.5s
def draw_text(cv2_img, txt):
    pil_img = Image.fromarray(cv2_img)
    draw = ImageDraw.Draw(pil_img)
    draw.text(txt_location, txt, fontColor, font=font)
    return np.array(pil_img)
    #return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    #cv2.putText(img, txt, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)

def load_model():
    # load config file
    with open(config_file) as data_file:
        config = json.load(data_file)
    model = C3D_Variant(config['num_classes'])
    model = torch.nn.DataParallel(model).to(device)
    #load chk
    if os.path.isfile(config['best_checkpoint']):
        checkpoint = torch.load(config['best_checkpoint'], map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> load model done!") 

    transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    train_data = VideoFolder(root=config['train_data_folder'],
                             csv_file_input=config['train_data_csv'],
                             csv_file_labels=config['labels_csv'],
                             clip_size=config['clip_size'],
                             nclips=1,
                             step_size=config['step_size'],
                             is_val=False,
                             transform=transform,
                             )

    print(" > Using {} processes for data loader.".format(
        config["num_workers"]))
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=config['batch_size'], shuffle=True,
        num_workers=config['num_workers'], pin_memory=True,
        drop_last=True)

    #predict loader
    predict_data = VideoFolder(root=config['predict_data_folder'],
                               csv_file_input=config['predict_data_csv'],
                               csv_file_labels=config['labels_csv'],
                               clip_size=config['clip_size'],
                               nclips=1,
                               step_size=config['step_size'],
                               is_val=True,
                               transform=transform,
                               predict_only=True)
    predict_loader = torch.utils.data.DataLoader(
         predict_data,
         batch_size=config['batch_size'], shuffle=False,
         num_workers=config['num_workers'], pin_memory=True,
         drop_last=False)
    
    assert len(train_data.classes) == config["num_classes"]
    return (predict_loader, model, train_data.classes_dict)  

def predict(predict_loader, model, class_to_idx=None):
    #switch model to test
    model.eval()
    logits_matrix = []
    with torch.no_grad():
        for i, (input, target) in enumerate(predict_loader):
            input = input.to(device)
            #predict
            output = model(input)          
            logits_matrix.append(output.detach().cpu().numpy())
            logits_matrix = np.concatenate(logits_matrix)
            predict = np.argmax(logits_matrix, axis=1) #get max element's index
            print(class_to_idx[predict[0]])#predict result

#====================client====================    

dataset_dir = "datasets/pending"
#idx = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 57, 60, 62, 65, 67, 70, 72]
idx = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 57]
# 写数据进程执行的代码: client
def client(q, p):
    print('Client Process : %s' % os.getpid())
    # camera config
    cam_id = 0
    video_height = 480
    video_width = 640
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height) #height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width) #width
    cap.set(cv2.CAP_PROP_FPS, 12) #fps
    print("initializing camera, please wait......")
    #load hand detect model
    detection_graph, sess = detector_utils.load_inference_graph()
    print("load hand detect model done!")
    tm.sleep(5)

    sample_rate = 12
    i = 0
    j = 0
    ready = False
    go = False
    drop = 5#drop first 5 and last 5
    predict_txt = "None"
    txt_showned_time = -1
    while True: 
        frames = []    
        ret, frame = cap.read()
        if p.full() or q.empty():
            if not p.empty():
                predict_txt = p.get()
                txt_showned_time = 0
                p.task_done()
                i = 0
                j = 0
            ready = True #ready to capture frame
        
        #draw txt
        if txt_showned_time != -1 and txt_showned_time <= diss_time:
            txted_frame = draw_text(frame, predict_txt)
            txt_showned_time = txt_showned_time + 1
        else:
            txted_frame = frame
            txt_showned_time = -1

        # show a frame
        cv2.imshow("capture", txted_frame)

        #hand detection trigger
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not go:
            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
            max = np.max(scores)
            if max > 0.1:
                go = True

        if ready and go:
            if i == idx[j] and j < len(idx):
                #print("======sample %dth frame=====" % j)
                q.put(Image.fromarray(np.uint8(cv2.resize(frame, (120, 100)))))
                j = j  + 1 
            if j >= len(idx):
                ready = False
                go = False
                j = 0#reset
            i = (i + 1) % 60    
        
        k = cv2.waitKey(1) & 0xFF
        #if k == ord('a'):
            #go = True
        #elif k == ord('q'):
            #break
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

#====================server==================== 
transform = Compose([
    CenterCrop(84),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225])
])
def server(q, p):
    print('Process to read: %s' % os.getpid())
    predict_loader, model, class_to_idx = load_model()
    print("model init done!")
    while True:
        imgs = [] #N*C*H*W
        for i in range(len(idx)):
            img = q.get(True) # H*W*C
            img = transform(img)
            imgs.append(torch.unsqueeze(img , 0))
        imgs = torch.cat(imgs)
        imgs = imgs.permute(1,0,2,3) # C*H*W
        input = torch.Tensor(imgs)
        input = torch.unsqueeze(input , 0)
        #print(input.shape)

        logits_matrix = []
        model.eval()
        with torch.no_grad():
            input = input.to(device)
            output = model(input)          
            logits_matrix.append(output)
            logits_matrix = np.concatenate(logits_matrix)
            predict = np.argmax(logits_matrix, axis=1) #get max element's index
            print(class_to_idx[predict[0]])#predict result
            print(label[predict[0]])#predict result
        if not p.full():
            p.put(label[predict[0]])#notify client
        else:
            p.join()#block

if __name__=='__main__':
    # 父进程创建Queue，并传给各个子进程：
    q = JoinableQueue(len(idx)) # 28 frames
    p = JoinableQueue(1)
    clientProcess = Process(target=client, args=(q, p))
    serverProcess = Process(target=server, args=(q, p))
    # 启动子进程pw，写入:
    clientProcess.start()
    # 启动子进程pr，读取:
    serverProcess.start()
    # 等待pw结束:
    clientProcess.join()
    # pr进程里是死循环，无法等待其结束，只能强行终止:
    serverProcess.terminate()