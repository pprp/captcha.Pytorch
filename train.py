from config.parameters import *
import torch as t
from torch import optim
from torch import nn
from lib.dataset import *
from torch.utils.data import DataLoader
import tqdm
from tools.Visdom import *
from torchnet import meter
from model.model import *
from lib.optimizer import RAdam,AdamW
import os


augTrainDataset = augCaptcha("./data/auged_train", train=True)
trainDataset = Captcha("./data/train/", train=True)
testDataset = Captcha("./data/test/", train=False)
augTrainDataLoader = DataLoader(augTrainDataset, batch_size=batchSize,
                                shuffle=True, num_workers=4)
trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
                                shuffle=True, num_workers=4)
testDataLoader = DataLoader(testDataset, batch_size=batchSize,
                            shuffle=True, num_workers=4)


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(model):
    avgLoss = 0.0
    if t.cuda.is_available():
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()
    # RAdam
    optimizer = RAdam(model.parameters(), lr=learningRate, betas=(0.9, 0.999), weight_decay=6.5e-4)
    #optimizer = optim.Adam(model.parameters(), lr=learningRate)
    vis = Visualizer(env = "ResCaptcha")
    loss_meter = meter.AverageValueMeter()
    best_acc = -1.
    for epoch in range(totalEpoch):
        print("="*13,"epoch:",epoch,"="*13)
        for circle, input in enumerate(trainDataLoader, 0):
            x, label = input
            # print('-'*5, x.size(), label.size())
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(x)
            # print(label1.size(),label2.size(),label3.size(),label4.size())
            # print(y1.shape, y2.shape, y3.shape, y4.shape)
            loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2) \
                , criterion(y3, label3), criterion(y4, label4)
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            if circle % printCircle == 0:
                print("step: %03d, Train loss is %.5f" % (circle, avgLoss / printCircle))
                # writeFile("step %d , Train loss is %.5f" % (circle, avgLoss / printCircle))
                vis.plot_many_stack({"train_loss": avgLoss})
                avgLoss = 0
                
        print("="*13,"aug epoch","="*13)
        for circle, input in enumerate(augTrainDataLoader, 0):
            x, label = input
            # print('-'*5, x.size(), label.size())
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(x)
            # print(label1.size(),label2.size(),label3.size(),label4.size())
            # print(y1.shape, y2.shape, y3.shape, y4.shape)
            loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2) \
                , criterion(y3, label3), criterion(y4, label4)
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            if circle % printCircle == 0:
                print("step: %03d, Train loss is %.5f" % (circle, avgLoss / printCircle))
                # writeFile("step %d , Train loss is %.5f" % (circle, avgLoss / printCircle))
                vis.plot_many_stack({"train_loss": avgLoss})
                avgLoss = 0
        if True:
            # one epoch once
            accuracy = test(model, testDataLoader)
            print("epoch: %03d, accuracy: %.3f" % (epoch, accuracy))
            vis.plot_many_stack({"test_acc":accuracy})
            if best_acc < accuracy:
                best_acc = accuracy
            if best_acc < accuracy or best_acc - accuracy < 0.01:
                model.save(str(epoch)+"_"+str(int(accuracy*1000)))


def test(model, testDataLoader):
    totalNum = len(os.listdir('./data/test'))
    rightNum = 0
    for circle, (x, label) in enumerate(testDataLoader, 0):
        label = label.long()
        if t.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        y1, y2, y3, y4 = model(x)
        small_bs = x.size()[0]# get the first channel
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(small_bs, 1), \
                         y2.topk(1, dim=1)[1].view(small_bs, 1), \
                         y3.topk(1, dim=1)[1].view(small_bs, 1), \
                         y4.topk(1, dim=1)[1].view(small_bs, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        diff = (y != label)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        rightNum += (small_bs - res)
    print(rightNum, totalNum)
    print("test acc: %s" % (float(rightNum) / float(totalNum)))
    return float(rightNum) / float(totalNum)


def writeFile(str):
    file = open("result.txt", "w")
    file.write(str)
    file.write("\n")
    file.flush()
    file.close()

if __name__ == '__main__':
    # net = RES50()
    # net = CaptchaNet()
    net = ResNet(ResidualBlock)
    net.loadIfExist("./model/net99_738.pth")
    train(net)
