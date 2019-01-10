import torch
import torch.nn as nn

class C3DDeepDeep(nn.Module):#Best Model 90%

    def __init__(self, num_classes):
        super(C3DDeepDeep, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) # temporal 18 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer1[0].weight, std=0.005)
        
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2)) # 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer2[0].weight, std=0.005)
        
        self.conv_layer3 = self._make_conv_layer(128, 128, (1, 2, 2), (1, 2, 2))# 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer3[0].weight, std=0.005)
        
        self.conv_layer4 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))# loss some temporal info 9
        nn.init.normal_(self.conv_layer4[0].weight, std=0.005)
        
        self.conv_layer5 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2)) # 4
        nn.init.normal_(self.conv_layer5[0].weight, std=0.005)
        
        self.conv_layer6 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))# 2
        nn.init.normal_(self.conv_layer6[0].weight, std=0.005)
        
        self.fc7 = nn.Linear(512, 1024)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(1024, 2048)
        nn.init.normal_(self.fc8.weight, std=0.005)
        self.fc8_act = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc9.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc7(x)      
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.fc8_act(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        return x

class C3DDeepDeep1(nn.Module): #保留更多的时序信息

    def __init__(self, num_classes):
        super(C3DDeepDeep1, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) # temporal 18 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer1[0].weight, std=0.005)
        
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2)) # 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer2[0].weight, std=0.005)
        
        self.conv_layer3 = self._make_conv_layer(128, 128, (1, 2, 2), (1, 2, 2))# 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer3[0].weight, std=0.005)
        
        self.conv_layer4 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))# loss some temporal info 9
        nn.init.normal_(self.conv_layer4[0].weight, std=0.005)
        
        self.conv_layer5 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2)) # 4
        nn.init.normal_(self.conv_layer5[0].weight, std=0.005)
        
        self.conv_layer6 = self._make_conv_layer(256, 512, (2, 2, 2), (2, 2, 2))# 2
        nn.init.normal_(self.conv_layer6[0].weight, std=0.005)
        
        self.fc7 = nn.Linear(1024, 2048)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(2048, 2048)
        nn.init.normal_(self.fc8.weight, std=0.005)
        self.fc8_act = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc9.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc7(x)      
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.fc8_act(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        return x

class C3DWithLessTimePooling(nn.Module): #减少时间维度的池化，保留更多的时序信息

    def __init__(self, num_classes):
        super(C3DWithLessTimePooling, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) #42 * 42 temporal 18 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer1[0].weight, std=0.005)
        
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2)) #21*21 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer2[0].weight, std=0.005)
        
        self.conv_layer3 = self._make_conv_layer(128, 128, (1, 2, 2), (1, 2, 2))#10*10 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer3[0].weight, std=0.005)
        
        self.conv_layer4 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))#5*5 loss some temporal info 9
        nn.init.normal_(self.conv_layer4[0].weight, std=0.005)
        
        self.conv_layer5 = self._make_conv_layer(256, 256, (1, 2, 2), (1, 2, 2))#2*2 # 9
        nn.init.normal_(self.conv_layer5[0].weight, std=0.005)
        
        self.conv_layer6 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))#1*1 4
        nn.init.normal_(self.conv_layer6[0].weight, std=0.005)
        
        self.fc7 = nn.Linear(1024, 1024)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(1024, 2048)
        nn.init.normal_(self.fc8.weight, std=0.005)
        self.fc8_act = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc9.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc7(x)      
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.fc8_act(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        return x

class C3DWithLessTimePooling_1(nn.Module): #减少时间维度的池化，保留更多的时序信息 底层的时间信息保留，高层的池化；

    def __init__(self, num_classes):
        super(C3DWithLessTimePooling_1, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) #42 * 42 temporal 18 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer1[0].weight, std=0.005)
        
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2)) #21*21 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer2[0].weight, std=0.005)
        
        self.conv_layer3 = self._make_conv_layer(128, 128, (1, 2, 2), (1, 2, 2))#10*10 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer3[0].weight, std=0.005)
        
        self.conv_layer4 = self._make_conv_layer(128, 256, (1, 2, 2), (1, 2, 2))#5*5 loss some temporal info 18
        nn.init.normal_(self.conv_layer4[0].weight, std=0.005)
        
        self.conv_layer5 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))#2*2 # 9
        nn.init.normal_(self.conv_layer5[0].weight, std=0.005)
        
        self.conv_layer6 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))#1*1 4
        nn.init.normal_(self.conv_layer6[0].weight, std=0.005)
        
        self.fc7 = nn.Linear(1024, 1024)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(1024, 2048)
        nn.init.normal_(self.fc8.weight, std=0.005)
        self.fc8_act = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc9.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc7(x)      
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.fc8_act(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        return x
    
class C3DDeepWithBN(nn.Module):#fc layer带bacth normal，同时也改变了

    def __init__(self, num_classes):
        super(C3DDeepWithBN, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) # temporal 18 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer1[0].weight, std=0.005)
        
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2)) # 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer2[0].weight, std=0.005)
        
        self.conv_layer3 = self._make_conv_layer(128, 128, (1, 2, 2), (1, 2, 2))# 前面的cnn层多保留时序信息 18
        nn.init.normal_(self.conv_layer3[0].weight, std=0.005)
        
        self.conv_layer4 = self._make_conv_layer(128, 256, (2, 2, 2), (2, 2, 2))# loss some temporal info 9
        nn.init.normal_(self.conv_layer4[0].weight, std=0.005)
        
        self.conv_layer5 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2)) # 4
        nn.init.normal_(self.conv_layer5[0].weight, std=0.005)
        
        self.conv_layer6 = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))# 2
        nn.init.normal_(self.conv_layer6[0].weight, std=0.005)
        
        self.fc7 = nn.Linear(512, 1024)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_batch_norm = nn.BatchNorm1d(1024)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(1024, 2048)
        nn.init.normal_(self.fc8.weight, std=0.005)
        self.fc8_batch_norm = nn.BatchNorm1d(2048)
        self.fc8_act = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc9.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc7(x) 
        x = self.fc7_batch_norm(x)
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.fc8_batch_norm(x)
        x = self.fc8_act(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        return x

class C3DDeepWithBigTemporalReceptiveField(nn.Module):
    #fc layer带bacth normal，输入变为28帧，且从第3层开始，增加时间维度的感受野大小，以捕获手势的动态性。20181226

    def __init__(self, num_classes):
        super(C3DDeepWithBigTemporalReceptiveField, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接 输入是28*84*84
        self.conv_layer1 = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) #temporal 28 前面的cnn层多保留时序信息 28
        nn.init.normal_(self.conv_layer1[0].weight, std=0.005)
        
        self.conv_layer2 = self._make_conv_layer(64, 128, (1, 2, 2), (1, 2, 2))#前面的cnn层多保留时序信息 28
        nn.init.normal_(self.conv_layer2[0].weight, std=0.005)
        
        self.conv_layer3 = self._make_conv_layer(128, 128, (2, 2, 2), (2, 2, 2))#temporal 14
        nn.init.normal_(self.conv_layer3[0].weight, std=0.005)
        
        self.conv_layer4 = self._make_conv_layer(128, 256, (2, 2, 2), (3, 2, 2))#temporal 5
        nn.init.normal_(self.conv_layer4[0].weight, std=0.005)
        
        self.conv_layer5 = self._make_conv_layer(256, 256, (2, 2, 2), (3, 2, 2)) #temporal 2
        nn.init.normal_(self.conv_layer5[0].weight, std=0.005)
        
        self.conv_layer6 = self._make_conv_layer(256, 256, (2, 2, 2), (3, 2, 2))#temporal 1
        nn.init.normal_(self.conv_layer6[0].weight, std=0.005)
        
        self.fc7 = nn.Linear(256, 512)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_batch_norm = nn.BatchNorm1d(512)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.fc8 = nn.Linear(512, 1024)
        nn.init.normal_(self.fc8.weight, std=0.005)
        self.fc8_batch_norm = nn.BatchNorm1d(1024)
        self.fc8_act = nn.ReLU()
        self.dropout8 = nn.Dropout(p=0.5)
        
        self.fc9 = nn.Linear(1024, num_classes)
        nn.init.normal_(self.fc9.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_c),
            nn.ReLU(),
            nn.MaxPool3d(pool_size, stride=stride, padding=0)
        )
        return conv_layer

    def forward(self, x):
        x = self.conv_layer1(x)
        x = self.conv_layer2(x)
        x = self.conv_layer3(x)
        x = self.conv_layer4(x)
        x = self.conv_layer5(x)
        x = self.conv_layer6(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc7(x) 
        x = self.fc7_batch_norm(x)
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.fc8(x)
        x = self.fc8_batch_norm(x)
        x = self.fc8_act(x)
        x = self.dropout8(x)
        
        x = self.fc9(x)
        return x

class C3D_Variant(nn.Module):#尝试一个更深的网络，受限于GPU Memory 因此做一些阉割变化；
    #fc layer带bacth normal，输入变为28帧，且从第3层开始，增加时间维度的感受野大小，以捕获手势的动态性。20181226

    def __init__(self, num_classes):
        super(C3D_Variant, self).__init__()
        #初始化网络结构
        #6层3D卷积 + 3层全连接 输入是28*84*84
        self.conv_1a = self._make_conv_layer(3, 64, (1, 2, 2), (1, 2, 2)) #temporal 28 前面的cnn层多保留时序信息 28*42*42
        nn.init.normal_(self.conv_1a[0].weight, std=0.005)
        
        self.conv_2a = self._make_conv_layer(64, 128, (2, 2, 2), (2, 2, 2))#14*21*21
        nn.init.normal_(self.conv_2a[0].weight, std=0.005)
        
        self.conv_3a = self._make_conv_layer(128, 128, None, None)#14*21*21
        nn.init.normal_(self.conv_3a[0].weight, std=0.005)
        self.conv_3b = self._make_conv_layer(128, 128, (2, 3, 3), (2, 3, 3))#7*7*7 
        nn.init.normal_(self.conv_3b[0].weight, std=0.005)
        
        self.conv_4a = self._make_conv_layer(128, 256, None, None) #7*7*7
        nn.init.normal_(self.conv_4a[0].weight, std=0.005)
        self.conv_4b = self._make_conv_layer(256, 256, (2, 2, 2), (2, 2, 2))#3*3*3
        nn.init.normal_(self.conv_4b[0].weight, std=0.005)
        
        self.conv_5a = self._make_conv_layer(256, 256, None, None) #3*5*5
        nn.init.normal_(self.conv_5a[0].weight, std=0.005)
        self.conv_5b = self._make_conv_layer(256, 256, (3, 3, 3), (3, 3, 3))#1*1*1
        nn.init.normal_(self.conv_5b[0].weight, std=0.005)
        
        self.fc6 = nn.Linear(256, 512)
        nn.init.normal_(self.fc6.weight, std=0.005)
        self.fc6_batch_norm = nn.BatchNorm1d(512)
        self.fc6_act = nn.ReLU()
        self.dropout6 = nn.Dropout(p=0.5)
        
        self.fc7 = nn.Linear(512, 512)
        nn.init.normal_(self.fc7.weight, std=0.005)
        self.fc7_batch_norm = nn.BatchNorm1d(512)
        self.fc7_act = nn.ReLU()
        self.dropout7 = nn.Dropout(p=0.5)
        
        self.softmax = nn.Linear(512, num_classes)
        nn.init.normal_(self.softmax.weight, std=0.01)

    def _make_conv_layer(self, in_c, out_c, pool_size, stride):
        layers = []
        layers.append(nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1))
        layers.append(nn.BatchNorm3d(out_c))
        layers.append(nn.ReLU())
        if pool_size and stride:
            layers.append(nn.MaxPool3d(pool_size, stride=stride, padding=0))
        conv_layer = nn.Sequential(*layers)
        return conv_layer

    def forward(self, x):
        x = self.conv_1a(x)
        
        x = self.conv_2a(x)
        
        x = self.conv_3a(x)
        x = self.conv_3b(x)
        
        x = self.conv_4a(x)
        x = self.conv_4b(x)
        
        x = self.conv_5a(x)
        x = self.conv_5b(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc6(x) 
        x = self.fc6_batch_norm(x)
        x = self.fc6_act(x)
        x = self.dropout6(x)
        
        x = self.fc7(x)
        x = self.fc7_batch_norm(x)
        x = self.fc7_act(x)
        x = self.dropout7(x)
        
        x = self.softmax(x)
        return x
    
if __name__ == "__main__":
    input_tensor = torch.autograd.Variable(torch.rand(5, 3, 18, 84, 84))
    model = C3DDeepDeep(27) #C3D(27).cuda()
    output = model(input_tensor) #model(input_tensor.cuda())
    print(output.size())