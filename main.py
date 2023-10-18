from mnn import network, layer, activations as a, dataset
from mnn.utils import mae_loss, list_avg

net = network()
lay = layer(2,2,a.relu)
net.add_layer(lay)
lay = layer(2,2,a.relu)
net.add_layer(lay)
lay = layer(2,1,a.straight)
net.add_layer(lay)

print(net)
data = dataset()

data.add_data([0,0],[0])
data.add_data([0,1],[1])
data.add_data([1,0],[1])
data.add_data([1,1],[0])

real, temp = net.run_all_data(data)

losses = []

for r,t in zip(real,temp):
  losses.append(mae_loss(r,t))

print(real)
print(temp)

print("\n")

print(losses)

print(list_avg(losses))

# net.save(r"network.txt")