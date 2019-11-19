from parameters import *
from model import *
from train import *

if __name__ == '__main__':
    net = RES50(ResidualBlock)
    # net = CaptchaNet()
    net.loadIfExist()
    train(net)
