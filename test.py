from model.model import *
from lib.dataset import *
from train import *


def userTest(model, dataLoader):
    totalNum = 0
    rightNum = 0
    badlist = []
    for circle, input in enumerate(dataLoader, 0):
        totalNum += 1
        x, label = input
        if t.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        realLabel = LabeltoStr([label[0][0], label[0][1], label[0][2], label[0][3]])
        # print(label,realLabel)
        y1, y2, y3, y4 = model(x)
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(1, 1), y2.topk(1, dim=1)[1].view(1, 1), \
                         y3.topk(1, dim=1)[1].view(1, 1), y4.topk(1, dim=1)[1].view(1, 1)
        y = t.cat((y1, y2, y3, y4), dim=1)
        # print(x,label,y)
        decLabel = LabeltoStr([y[0][0], y[0][1], y[0][2], y[0][3]])
        print("real: %s -> %s , %s" % (realLabel, decLabel, str(realLabel == decLabel)))
        if realLabel == decLabel:
            rightNum += 1
        else:
            badlist.append([realLabel,decLabel])

    for itm in badlist:
        print("False: ", itm[0], "=>", itm[1])
    print("\n total %s, right %s, wrong %s." % (totalNum, rightNum, totalNum-rightNum))

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description="weight path")
    parser.add_argument('--weight_path', type=str,default="./model/resNet10.pth")
    parser.add_argument('--test_path', type=str, default="./data/test")
    args = parser.parse_args()


    model = ResNet(ResidualBlock)
    model.eval()
    model.loadIfExist(args.weight_path)
    if t.cuda.is_available():
        model = model.cuda()
    userTestDataset = Captcha(args.test_path, train=True)
    userTestDataLoader = DataLoader(userTestDataset, batch_size=1,
                                    shuffle=True, num_workers=1)
    userTest(model, userTestDataLoader)
