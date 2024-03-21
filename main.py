from mnn import *
from mnn import activations as a
import time


net = network()

net.add_layer(layer(3,5,a.straight))
net.add_layer(layer(5,5,a.straight))
net.add_layer(layer(5,3,a.straight))

# s = save(net)
# s.save("test.mnn")


data = dataset()


data.add_data([1,2,3],[4,5,6])
data.add_data([2,3,4],[5,6,7])

def train_func(net:network, epoch:int, train:trainer):
    print(f"EPOCH: {epoch} LOSS: {train.get_test_data_loss()}")

train = trainer(net, 10, data, train_func)


# train = trainer(net, 10, data)

start = time.time()

train.start_train()

end = time.time()


print(f"TIME: {end-start} seconds")