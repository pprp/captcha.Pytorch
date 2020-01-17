from config.parameters import *
import torch as t
import torch
from torch import optim
from torch import nn
import torch.optim as optim
from lib.dataset import *
from torch.utils.data import DataLoader
import tqdm
from utils.Visdom import *
from torchnet import meter
from model.model import *
from lib.optimizer import RAdam, AdamW
import os
from model.dense import dense121
from model.senet import senet
from model.res18 import res18
from model.dualpooling import DualResNet
from model.BNNeck import bnneck
from lib.center_loss import CenterLoss
from model.IBN import res_ibn
from lib.scheduler import GradualWarmupScheduler

torch.manual_seed(42)
# import adabound

augTrainDataset = augCaptcha(augedTrainPath, train=True)
trainDataset = Captcha(trainPath, train=True)
testDataset = Captcha(testPath, train=False)
augTrainDataLoader = DataLoader(augTrainDataset,
                                batch_size=batchSize,
                                shuffle=True,
                                num_workers=4)
trainDataLoader = DataLoader(trainDataset,
                             batch_size=batchSize,
                             shuffle=True,
                             num_workers=4)
testDataLoader = DataLoader(testDataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=1)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ratio_c = 1
ratio_x = 1


def train_with_center(model):
    model.train()
    if torch.cuda.is_available():
        model = model.cuda()

    criterion_xent = nn.CrossEntropyLoss()
    criterion_cent = CenterLoss(num_classes=62, feat_dim=62)
    optimizer_centloss = optim.SGD(criterion_cent.parameters(), lr=0.005)
    # params = list(criterion_cent.parameters())+list(model.parameters())
    optimizer_model = optim.Adam(model.parameters(), lr=3e-4)
    # optimizer = RAdam(model.parameters(), lr=learningRate,
    #                   betas=(0.9, 0.999), weight_decay=6.5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learningRate)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6) # Cosine需要的初始lr比较大1e-2,1e-3都可以
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    # milestone_list = [10 * k for k in range(1, totalEpoch//10)]
    # scheduler = optim.lr_scheduler.MultiStepLR(  # lr 3e-3 best
    #     optimizer_model, milestones=milestone_list, gamma=0.5)
    scheduler = optim.lr_scheduler.StepLR(  # best lr 1e-3
        optimizer_model, step_size=20, gamma=0.5)

    vis = Visualizer(env="centerloss")
    loss_meter = meter.AverageValueMeter()
    avgLoss = 0.0
    loss_x_meter = meter.AverageValueMeter()
    loss_c_meter = meter.AverageValueMeter()

    best_acc = -1.
    for epoch in range(totalEpoch):
        loss_meter.reset()
        loss_x_meter.reset()
        loss_c_meter.reset()
        record_circle = 0
        for circle, input in enumerate(trainDataLoader, 0):
            record_circle = circle
            x, label = input
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2, label3, label4 = label[:,
                                                   0], label[:,
                                                             1], label[:,
                                                                       2], label[:,
                                                                                 3]
            y1, y2, y3, y4 = model(x)
            ####################################################################
            loss_x1, loss_x2 = criterion_xent(y1, label1), criterion_xent(
                y2, label2)
            loss_x3, loss_x4 = criterion_xent(y3, label3), criterion_xent(
                y4, label4)
            loss_x = loss_x1 + loss_x2 + loss_x3 + loss_x4
            ####################################################################
            loss_c1, loss_c2 = criterion_cent(y1, label1), criterion_cent(
                y2, label2)
            loss_c3, loss_c4 = criterion_cent(y3, label3), criterion_cent(
                y4, label4)
            loss_c = loss_c1 + loss_c2 + loss_c3 + loss_c4
            ####################################################################
            loss = ratio_c * loss_c + ratio_x * loss_x
            ####################################################################
            loss_c_meter.add(loss_c.item())
            loss_x_meter.add(loss_x.item())
            loss_meter.add(loss.item())
            optimizer_centloss.zero_grad()
            optimizer_model.zero_grad()
            ####################################################################
            loss.backward()
            optimizer_model.step()
            ####################################################################
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / ratio_c)
            optimizer_centloss.step()
            ####################################################################
            if circle % printCircle == 0:
                print(
                    "epoch:%02d step: %03d train loss:%.5f model loss:%.2f center loss:%.2f"
                    % (epoch, circle, loss_meter.value()[0],
                       loss_x_meter.value()[0], loss_c_meter.value()[0]))
                # writeFile("step %d , Train loss is %.5f" % (circle, avgLoss / printCircle))
                vis.plot_many_stack({
                    "train_loss": loss_meter.value()[0],
                    "model loss": loss_x_meter.value()[0],
                    "center loss": loss_c_meter.value()[0]
                })
                loss_meter.reset()
                loss_c_meter.reset()
                loss_x_meter.reset()

        for circle, input in enumerate(augTrainDataLoader, record_circle):
            x, label = input
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2 = label[:, 0], label[:, 1]
            label3, label4 = label[:, 2], label[:, 3]
            y1, y2, y3, y4 = model(x)
            ####################################################################
            loss_x1, loss_x2 = criterion_xent(y1, label1), criterion_xent(
                y2, label2)
            loss_x3, loss_x4 = criterion_xent(y3, label3), criterion_xent(
                y4, label4)
            loss_x = loss_x1 + loss_x2 + loss_x3 + loss_x4
            ####################################################################
            loss_c1, loss_c2 = criterion_cent(y1, label1), criterion_cent(
                y2, label2)
            loss_c3, loss_c4 = criterion_cent(y3, label3), criterion_cent(
                y4, label4)
            loss_c = loss_c1 + loss_c2 + loss_c3 + loss_c4
            ####################################################################
            loss = ratio_c * loss_c + ratio_x * loss_x
            ####################################################################
            optimizer_centloss.zero_grad()
            optimizer_model.zero_grad()
            ####################################################################
            loss_c_meter.add(loss_c.item())
            loss_x_meter.add(loss_x.item())
            loss_meter.add(loss.item())
            avgLoss += loss.item()
            loss.backward()
            optimizer_model.step()
            ####################################################################
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / ratio_c)
            optimizer_centloss.step()
            ####################################################################
            if circle % printCircle == 0:
                print(
                    "epoch:%02d step: %03d train loss:%.5f model loss:%.2f center loss:%.2f"
                    % (epoch, circle, loss_meter.value()[0],
                       loss_x_meter.value()[0], loss_c_meter.value()[0]))
                vis.plot_many_stack({
                    "train_loss": loss_meter.value()[0],
                    "model loss": loss_x_meter.value()[0],
                    "center loss": loss_c_meter.value()[0]
                })
                loss_meter.reset()
                loss_x_meter.reset()
                loss_c_meter.reset()
        if True:
            # one epoch once
            scheduler.step()
            accuracy = test(model, testDataLoader)
            print("Learning rate: %.10f" % (scheduler.get_lr()[0]))
            print("epoch: %03d, accuracy: %.3f" % (epoch, accuracy))
            vis.plot_many_stack({"test_acc": accuracy})
            if best_acc < accuracy:
                best_acc = accuracy
            if best_acc < accuracy or best_acc - accuracy < 0.01:
                model.save(str(epoch) + "_" + str(int(accuracy * 1000)))


def train_original(model):
    vis = Visualizer(env="old one")
    model.train()
    avgLoss = 0.0
    if torch.cuda.is_available():
        model = model.cuda()
    criterion = nn.CrossEntropyLoss()
    #optimizer = adabound.AdaBound(model.parameters(), lr=learningRate, final_lr=1e-5, gamma=1e-4)
    # RAdam
    optimizer = RAdam(model.parameters(),
                      lr=learningRate,
                      betas=(0.9, 0.999),
                      weight_decay=6.5e-4)
    # optimizer = optim.Adam(model.parameters(), lr=learningRate)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6) # Cosine需要的初始lr比较大1e-2,1e-3都可以
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
    scheduler_after = optim.lr_scheduler.StepLR(optimizer,
                                                step_size=20,
                                                gamma=0.5)
    scheduler = GradualWarmupScheduler(optimizer,
                                       8,
                                       10,
                                       after_scheduler=scheduler_after)
    # milestone_list = [10 * k for k in range(1, totalEpoch//10)]
    # scheduler = optim.lr_scheduler.MultiStepLR(  # lr 3e-3 best
    #     optimizer, milestones=milestone_list, gamma=0.5)

    loss_meter = meter.AverageValueMeter()
    best_acc = -1.
    for epoch in range(totalEpoch):
        record_circle = 0
        for circle, input in enumerate(trainDataLoader, 0):
            record_circle = circle
            x, label = input
            # print('-'*5, x.size(), label.size())
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2 = label[:, 0], label[:, 1]
            label3, label4 = label[:, 2], label[:, 3]
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(x)
            # print(label1.size(),label2.size(),label3.size(),label4.size())
            # print(y1.shape, y2.shape, y3.shape, y4.shape)
            loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(
                y2, label2), criterion(y3, label3), criterion(y4, label4)
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            if circle % printCircle == 0:
                print("epoch:%02d | step: %03d | Train loss is %.5f" %
                      (epoch, circle, avgLoss / printCircle))
                vis.plot_many_stack({"train_loss": avgLoss})
                avgLoss = 0

        # print("="*13, "aug epoch", "="*13)
        for circle, input in enumerate(augTrainDataLoader, record_circle):
            x, label = input
            # print('-'*5, x.size(), label.size())
            if torch.cuda.is_available():
                x = x.cuda()
                label = label.cuda()
            label = label.long()
            label1, label2 = label[:, 0], label[:, 1]
            label3, label4 = label[:, 2], label[:, 3]
            optimizer.zero_grad()
            y1, y2, y3, y4 = model(x)
            # print(label1.size(),label2.size(),label3.size(),label4.size())
            # print(y1.shape, y2.shape, y3.shape, y4.shape)
            loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(
                y2, label2), criterion(y3, label3), criterion(y4, label4)
            loss = loss1 + loss2 + loss3 + loss4
            loss_meter.add(loss.item())
            # print(loss)
            avgLoss += loss.item()
            loss.backward()
            optimizer.step()
            if circle % printCircle == 0:
                print("epoch:%02d | step: %03d | Train loss is %.5f" %
                      (epoch, circle, avgLoss / printCircle))
                vis.plot_many_stack({"train_loss": avgLoss})
                avgLoss = 0
        if True:
            # one epoch once
            scheduler.step()
            accuracy = test(model, testDataLoader)
            print("Learning rate: %.10f" % (scheduler.get_lr()[0]))
            print("epoch: %03d, accuracy: %.3f" % (epoch, accuracy))
            vis.plot_many_stack({"test_acc": accuracy})
            if best_acc < accuracy:
                best_acc = accuracy
            if best_acc < accuracy or best_acc - accuracy < 0.01:
                model.save(str(epoch) + "_" + str(int(accuracy * 1000)))


def test(model, testDataLoader):
    model.eval()
    totalNum = len(os.listdir('./data/test'))
    rightNum = 0
    sum_loss = 0
    criterion = nn.CrossEntropyLoss()
    for circle, (x, label) in enumerate(testDataLoader, 0):
        label = label.long()
        if torch.cuda.is_available():
            x = x.cuda()
            label = label.cuda()
        y1, y2, y3, y4 = model(x)
        label1, label2 = label[:, 0], label[:, 1]
        label3, label4 = label[:, 2], label[:, 3]
        loss1, loss2, loss3, loss4 = criterion(y1, label1), criterion(
            y2, label2), criterion(y3, label3), criterion(y4, label4)
        loss = loss1 + loss2 + loss3 + loss4

        small_bs = x.size()[0]  # get the first channel
        y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(small_bs, 1), \
            y2.topk(1, dim=1)[1].view(small_bs, 1), \
            y3.topk(1, dim=1)[1].view(small_bs, 1), \
            y4.topk(1, dim=1)[1].view(small_bs, 1)
        y = torch.cat((y1, y2, y3, y4), dim=1)
        diff = (y != label)
        diff = diff.sum(1)
        diff = (diff != 0)
        res = diff.sum(0).item()
        rightNum += (small_bs - res)
        # sum_loss += loss
    print(rightNum, totalNum)
    print("test acc: %s" % (float(rightNum) / float(totalNum)))
    # , sum_loss / float(len(testDataLoader.dataset))
    return float(rightNum) / float(totalNum)


if __name__ == '__main__':
    # net = RES50()
    # net = CaptchaNet()
    net = ResNet(ResidualBlock)
    # net = dense121()
    # net = senet()
    # net = res18()
    # net = DualResNet(ResidualBlock)
    # net = bnneck()
    # net = res_ibn() # ibn block do not improve
    # net.load_model("./weights/senet_new.pth")
    # net.load_model("./model/net99_738.pth")
    # train(net)
    # train_with_center(net)
    train_original(net)
