import os

def test(model, testDataLoader):
    x, label = input
    label = label.long()
    y1, y2, y3, y4 = model(x)
    y1, y2, y3, y4 = y1.topk(1, dim=1)[1].view(testSize, 1), y2.topk(1, dim=1)[1].view(testSize, 1), \
                     y3.topk(1, dim=1)[1].view(testSize, 1), y4.topk(1, dim=1)[1].view(testSize, 1)
    y = t.cat((y1, y2, y3, y4), dim=1)
    diff = (y != label)
    diff = diff.sum(1)
    diff = (diff != 0)
    res = diff.sum(0)[0]
    print("the accuracy of test set is %s" % (str(float(rightNum) / float(totalNum))))

fileList = os.listdir("./model/captcha/")
print(fileList)
