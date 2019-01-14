from multiprocessing import Process, JoinableQueue
import os, time, random
import argparse
import os
import sys
import shutil
import json
import glob
import pickle
import csv
import numpy as np
import zlib
from time import time
import time as tm
import cv2
from PIL import Image, ImageDraw, ImageFont

from utils import detector_utils as detector_utils
import tensorflow as tf
import datetime
import socket
import zlib
import math
from context.context import Context
from callbacks.callback import FengHuoCallBack
from protocol.predict_protocol import SimplePredictionProtocol

label = ["向左划","向右划","向下划","向上划","手推向远处","手从远处拉回","两根手指向左滑动","两根手指向右滑动","两根手指向下滑动",
"两根手指向上滑动","两根手指推向远处","两根手指从远处拉回","手向前滚动","手向后滚动","顺时针转手","逆时针转手","用手放大","用手缩小",
"两根手指放大","两根手指缩小","拇指向上","拇指向下","摇动手掌","停止","抖动手指","没有手势","其他手势"]

font = ImageFont.truetype("simhei.ttf", 20, encoding="utf-8")
ascent, descent = font.getmetrics()
small_font = ImageFont.truetype("simhei.ttf", 14, encoding="utf-8")
fontScale = 1
fontColor = (255,0,0)
lineType = 2
diss_time = 75 # about 2.5s
line_colors = [(0xFF, 0x00, 0x00), (0x30, 0x30, 0xFF), (0x00, 0xFF, 0x00), (0xCD, 0x26, 0x7D), (0x22, 0x22, 0xB2)]
anchor = (440, 60)
#prob line config
base_line = 300#the length of line while probability is 100%
margin_top = 10
line_width = 20
margin_left = 20

def draw_text(cv2_img, txt, pred_top5=None):
    h = cv2_img.shape[0]
    w = cv2_img.shape[1]
    #calculate other anchor points.
    anchor = (0.5*w, 0.05*h)
    base_line = (w-anchor[0])*0.7
    line_width = h // 36
    margin_left = w // 64
    margin_top = margin_left // 2
    pil_img = Image.fromarray(cv2_img)
    draw = ImageDraw.Draw(pil_img)
    #draw prediction@1 text
    draw.text(anchor, str(txt), fontColor, font=font)
    #draw pred@5 bar
    if not pred_top5 == None:
        y = 1.5*margin_top + anchor[1] + ascent + descent
        x = anchor[0]
        for i in range(5):
            pred = pred_top5[i]
            color = line_colors[i]
            x_line_end = x + base_line*pred['prob']
            draw.line([(x, y),(x_line_end, y)], fill=color, width=line_width)
            draw.text((x_line_end + margin_left, y), "{:.2f}%  {}".format(pred['prob']*100, label[pred['label']]), color, font=small_font)
            y = y + margin_top + line_width
    return np.array(pil_img)


#====================UI process====================
def ui(frm_q, ret_q, idx, args, ctx):
    print('UI Process : %s' % os.getpid())
    # camera config
    cam_id = args.camera_id
    video_height = args.video_height
    video_width = args.video_width
    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_height) #height
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_width) #width
    print("[UI] > initializing camera, please wait......")
    tm.sleep(5)
    print("[UI] > camera initialization done.")

    #load hand detect model
    hand_threshold = args.hand_threshold # hand detection threshold
    print("[UI] > hand detection model loading......")
    detection_graph, sess = detector_utils.load_inference_graph()
    print("[UI] > hand detection model loaded, using threshold: %2f." % hand_threshold)

    i = 0
    j = 0
    ready = False
    go = False
    predict_txt = "None"
    txt_showned_time = -1
    init = True

    while True:
        frames = []
        ret, frame = cap.read()
        if init:
            ready = True
            init = False
        elif ret_q.full() and frm_q.empty():
            simple_protocol = ret_q.get()
            if simple_protocol == None:
                print("[UI] > background process has occurred fatal error.")
                sys.exit(1)
            for callback in ctx.callbacks:
                callback.onPredictTop1(None, simple_protocol.pred[0])
                callback.onPredictTop5(None, simple_protocol.pred)
            predict_txt = label[simple_protocol.pred[0]["label"]]
            txt_showned_time = 0
            ret_q.task_done()
            i = 0
            j = 0
            ready = True #ready to capture frame

        #draw txt in camera capture windows
        if txt_showned_time != -1 and txt_showned_time <= diss_time:
            txted_frame = draw_text(frame, predict_txt, simple_protocol.pred)
            txt_showned_time = txt_showned_time + 1
        else:
            txted_frame = frame
            txt_showned_time = -1

        # show a frame in windows
        cv2.imshow("FengHuo Gesture Recognition System", txted_frame)
        #hand detection trigger
        image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if not go and ready:
            boxes, scores = detector_utils.detect_objects(image_np, detection_graph, sess)
            max = np.max(scores)
            if max > hand_threshold:
                for callback in ctx.callbacks:
                    callback.onHandIn(None)
                go = True

        if ready and go:
            if i == idx[j] and j < len(idx):
                for callback in ctx.callbacks:
                    callback.onFrameSampled(None, frame)
                frm_q.put(Image.fromarray(np.uint8(cv2.resize(frame, (120, 100)))))
                j = j  + 1
            if j >= len(idx):
                ready = False
                go = False
                j = 0#reset
            i = (i + 1) % (math.floor(idx[-1]) + 1)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


#====================background process====================
def background(frm_q, ret_q, host, port, idx):
    print('background Process : %s' % os.getpid())
    sock = socket.socket()
    sock.setsockopt(socket.SOL_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
    except socket.error as e:
        print("[background] > connection error caused by %s" % e)
        if ret_q.empty():
            ret_q.put(-999)
        sys.exit(1)
    print("[background] > connect to %s:%2d" % (host, port))
    while True:
        frames = []
        for i in range(len(idx)):
            frame = frm_q.get(True)
            print("[background] > send %d th frame" % i)
            frames.append(frame)
        ret = sock.send(zlib.compress(pickle.dumps(frames)))
        if ret == 0:
            sock.send("None")
            raise RuntimeError("socket connection broken")
        #receive data from remote GPU server
        recv = sock.recv(65535) #SimplePredictionProtocol:
        print("recive byte length : %d" % (len(recv)))
        if not recv:
            sock.send("None")
            raise RuntimeError("socket connection broken")
        simple_protocol = pickle.loads(zlib.decompress(recv), encoding='bytes')
        print(type(simple_protocol))
        print(simple_protocol.pred)
        predict = simple_protocol.pred[0]["label"]
        prob = simple_protocol.pred[0]["prob"]
        print("[background] > predict@1 label:%s[%d]--%.4f" % (label[predict],predict, prob))
        if not ret_q.full():
            ret_q.put(simple_protocol)#notify UI
        else:
            ret_q.join()#block


def arg_parse():
    parser = argparse.ArgumentParser(description='FengHuo Hand Gesture client side.')
    parser.add_argument('--hand_threshold', '-t', type=float, default=0.1,
            help='hand detection threshold.')
    parser.add_argument('--camera_id', '-n', type=int, default=0, help='id of camera for use.')
    parser.add_argument('--video_height', '-y', type=int, default=480,
            help='camera capture height.')
    parser.add_argument('--video_width', '-x', type=int, default=640,
            help='camera capture width.')
    parser.add_argument('--port', '-p', type=int, default=9999, help='port to connect.')
    parser.add_argument('--host', '-H', default="127.0.0.1",
            help="host to connect.")
    parser.add_argument('--frm_number', '-f', type=int, default=26,
            help="number of frame sampled by web camera.")
    args = parser.parse_args()
    if args.frm_number <= 0 or args.frm_number > 30:
        parser.print_help()
        sys.exit(1)
    return args


class SimpleFengHuoCallBack(FengHuoCallBack):
    def onHandIn(self, ctx):
        self.print("Hands Come In.")

    def onPredictTop1(self, ctx, prediction_top1):
        self.print("pred@1: {:02d}:{:.2f}%".format(prediction_top1["label"], 100*prediction_top1["prob"]))

    def onPredictTop5(self, ctx, prediction_top5):
        s = "pred@5\n"
        for pred in prediction_top5:
            s += "{:02d}:{:.2f}%\n".format(pred["label"], 100*pred['prob'])
        self.print(s)

    def onFrameSampled(self, ctx, frm):
        self.print("Frame Sampled.")

    def print(self, str):
        print("[simple_fenghuo_callback] > " + str)


frm_idx = [0, 2, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 47, 50, 52, 55, 57, 60, 62, 65, 67, 70, 72] #30frm
def main():
    args = arg_parse()
    frm_cnt = args.frm_number
    host = args.host
    port = args.port
    idx = frm_idx[0:frm_cnt]
    assert len(idx) == frm_cnt

    #global context
    simple_callback = SimpleFengHuoCallBack()
    ctx = Context()
    ctx.register(simple_callback)

    frm_q = JoinableQueue(len(idx)) #queue to transfer frame data.
    ret_q = JoinableQueue(1)#queue used to send predict result to ui process.
    uiProcess = Process(target=ui, args=(frm_q, ret_q, idx, args, ctx))
    backgroundProcess = Process(target=background, args=(frm_q, ret_q, host, port, idx))
    uiProcess.start()
    backgroundProcess.start()
    # 等待uiProcess结束:
    uiProcess.join()
    backgroundProcess.terminate()


if __name__=='__main__':
    main()
