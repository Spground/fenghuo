import pickle
import torch
import torch.nn as nn
from c3d_model_deep_deep import *
from torchvision.transforms import *
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from twisted.internet.protocol import Protocol, Factory
import json
import os
import zlib
import sys
import argparse


config_file =  "configs/model_config.json"
def load_model(device):
    # load config file
    with open(config_file) as data_file:
        config = json.load(data_file)
    model = C3D_Variant(config['num_classes'])
    model = torch.nn.DataParallel(model).to(device)
    #load chk
    if os.path.isfile(config['best_checkpoint']):
        checkpoint = torch.load(config['best_checkpoint'], map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
    return model


class RecognitionProtocol(Protocol):
    received_data = b''
    factory = None
    transform = Compose([
        CenterCrop(84),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225])
    ])

    def __init__(self, factory):
        self.factory = factory

    def dataReceived(self, data):
        if data == b'None':
            self.received_data = b''
            return
        try:
            self.received_data += data
            decompressed_data = zlib.decompress(self.received_data)
            frames = pickle.loads(decompressed_data, encoding="bytes")
            print("recv %d frame" % (len(frames)))
        except zlib.error:
            return
        self.received_data = b''
        if len(frames) == self.factory.frm_cnt:
            #begin to predict
            pred = self.predict(frames)
            self.transport.write(str(pred).encode("UTF-8"))


    def connectionMade(self):
        print("connection made!")

    def connectionLost(self, reason):
        print("connection lost!")

    def handle_data(self, data, *more):
        self.transport.write(data)

    def predict(self, frames):
        imgs = []
        for frm in frames:
            frm = self.transform(frm)
            imgs.append(torch.unsqueeze(frm , 0))
        imgs = torch.cat(imgs)
        imgs = imgs.permute(1,0,2,3) # C*H*W
        input = torch.Tensor(imgs)
        input = torch.unsqueeze(input , 0)
        logits_matrix = []
        self.factory.model.eval()
        with torch.no_grad():
            input = input.to(self.factory.device)
            output = self.factory.model(input)
            logits_matrix.append(output.detach().cpu().numpy())
            logits_matrix = np.concatenate(logits_matrix)
            predict = np.argmax(logits_matrix, axis=1) #get max element's index
        print(predict[0])
        return predict[0]


class RecognitionFactory(Factory):
    model = None
    frm_cnt = 26
    device = None

    def __init__(self, model, device, frm_cnt=26):
        self.model = model
        self.frm_cnt = frm_cnt
        self.device = device

    def buildProtocol(self, addr):
        return RecognitionProtocol(self)

str2bool = lambda x: (str(x).lower == 'true')
def arg_parse():
    parser = argparse.ArgumentParser(
        description='fenghuo gesture recognition server side.')
    parser.add_argument('--port', '-p', type=int, default=9999, help='port listened on')
    parser.add_argument('--host', '-H', default="0.0.0.0", help='host listened on')
    parser.add_argument('--gpu', '-g', default=0, type=int,
                        help="gpu id for use")
    parser.add_argument('--use_gpu', default=True, type=str2bool, help="flag to use gpu or not")

    parser.add_argument('--frm_number', '-f', type=int, default=26,
                        help="number of frames used to predcit")
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()
    host = args.host
    port = args.port
    if args.use_gpu:
        gpu = args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    frm_cnt = args.frm_number
    from twisted.internet import reactor, endpoints
    from twisted.internet.endpoints import TCP4ServerEndpoint
    device = torch.device("cuda" if args.use_gpu else "cpu")
    print("model loading......")
    model = load_model(device)
    print("model loaded!")
    factory = RecognitionFactory(model, device, frm_cnt)
    endpoint = TCP4ServerEndpoint(reactor, port)
    endpoint.listen(factory)
    print("listen on %s:%d" % (host, port))
    reactor.run()

if __name__ == '__main__':
    main()
