from parameters import *
import torch as t
from torch import optim
from torch import nn
from dataset import *
from torch.utils.data import DataLoader
import tqdm
from Visdom import *
from torchnet import meter



def train(model):
    avgLoss = 0.0
    if t.cuda.is_available():
        model = model.cuda()
    trainDataset = Captcha("./data/train/", train=True)
    testDataset = Captcha("./data/test/", train=False)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize,
                                 shuffle=True, num_workers=4)
    testDataLoader = DataLoader(testDataset, batch_size=batchSize,
                                shuffle=True, num_workers=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningRate)
    vis = Visualizer(env = "ResCaptcha")
    loss_meter = meter.AverageValueMeter()
    for epoch in range(totalEpoch):
        for circle, input in tqdm.tqdm(enumerate(trainDataLoader, 0)):
            x, label = input
            if t.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2, label3, label4 = label[:, 0], label[:, 1], label[:, 2], label[:, 3]
            # print(label1,label2,label3,label4)
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(x)
            # print(y1.shape, y2.shape, y3.shape, y4.shape)
            loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(y2, label2) \
                , criterion(y3, label3), criterion(y4, label4)
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            if circle % printCircle == 1:
                print("after %d circle,the train loss is %.5f" %
                      (circle, avgLoss / printCircle))
                writeFile("after %d circle,the train loss is %.5f" %
                          (circle, avgLoss / printCircle))
                vis.plot_many_stack({"train_loss": avgLoss})
                avgLoss = 0
            if circle % testCircle == 1:
                accuracy = test(model, testDataLoader)
                vis.plot_many_stack({"test_acc":accuracy})
            if circle % saveCircle == 1:
                model.save(str(epoch)+"_"+str(saveCircle))


def test(model, testDataLoader):
    totalNum = testNum * batchSize
    rightNum = 0
    for circle, input in enumerate(testDataLoader, 0):
        if circle >= testNum:
            break
        x, label = input
        label = label.long()
        if t.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        y1, y2, y3, y4 = model(x)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(batchSize, 1), y2.topk(1, dim=1)[1].view(batchSize, 1), \
                         y3.topk(1, dim=1)[1].view(batchSize, 1), y4.topk(1, dim=1)[1].view(batchSize, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        diff = (y != label)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        rightNum += (batchSize - res)
    print("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))
    writeFile("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))
    return float(rightNum) / float(totalNum)


def writeFile(str):
    file = open("result.txt", "a+")
    file.write(str)
    file.write("\n\n")
    file.flush()
    file.close()
