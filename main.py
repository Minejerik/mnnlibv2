from mnn import *
from mnn import activations as a


net = network()

net.add_layer(layer(3,5,a.straight))
net.add_layer(layer(5,5,a.straight))
net.add_layer(layer(5,3,a.straight))

s = save(net)
s.save("test.mnn")