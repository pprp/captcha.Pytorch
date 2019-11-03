from parameters import *
from model import *
from train import *

if __name__ == '__main__':
    net = ResNet(ResidualBlock)
    # net = CaptchaNet()
    net.loadIfExist()
    train(net)
