# coding: utf-8
from decodeMinist import *
from nueralnet import *

train_images = decode_idx3_ubyte(train_images_idx3_ubyte_file)
train_labels = decode_idx1_ubyte(train_labels_idx1_ubyte_file)
test_images = decode_idx3_ubyte(test_images_idx3_ubyte_file)
test_labels = decode_idx1_ubyte(test_labels_idx1_ubyte_file)
trainingimages = [(im / 255).reshape(1, 784) for im in train_images]  # 归一化
traininglabels = [vectorized_result(int(i)) for i in train_labels]
testimages = [(im / 255).reshape(1, 784) for im in test_images]
testlabels = [l for l in test_labels]

print(type(traininglabels[0][0][0]))
net = NueraLNet([28 * 28, 30, 15, 10])
net.train_net(trainingimages, traininglabels, 30, 5, 10, testimages, testlabels)
# net.save_training()
# net.read_training()
# net.test_net(testimages, testlabels)
print("end")
