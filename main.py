from mnn import network, layer, activations as a, dataset,trainer
from mnn.utils import mae_loss, list_avg

net = network()
lay = layer(2,2,a.relu)
net.add_layer(lay)
lay = layer(2,4,a.relu)
net.add_layer(lay)
lay = layer(4,1,a.straight)
net.add_layer(lay)

print(net)
data = dataset()

data.add_data([0,0],[0])
data.add_data([0,1],[1])
data.add_data([1,0],[1])
data.add_data([1,1],[0])

train = trainer(net,75,data)


print(train.get_full_data_loss())

train.start_train()

print(train.get_full_data_loss())


net = train.get_net()

print(net.run_all_data(data))
# net.save(r"network.txt")