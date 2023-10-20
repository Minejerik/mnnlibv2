from mnn import network, layer, activations as a, dataset, trainer, save, load
from mnn.loader import load_csv
from random import choice

# # create network
# net = network()
# # create input layer w/ relu activation function
# # it has 2 input and 4 output nodes
# lay = layer(2,15,a.relu)
# net.add_layer(lay)
# # create hidden layer also w/ relu activation function
# # this time w/ 4 input nodes and 4 output nodes
# lay = layer(15,15,a.leakyrelu)
# net.add_layer(lay)
# # create output layer w/ straight activation function
# # it has 4 input nodes and 1 output node
# lay = layer(15,1,a.straight)
# net.add_layer(lay)

l = load("temp.mnn")
net = l.load()


#print the network
#allows for easy viewing
#most classes can be printed
print(net)

#create dataset
# data = load_csv("tested.csv")
data = dataset()
data.add_data([0,0],[0])
data.add_data([0.5,0.5],[1])
data.add_data([0.5,1],[1.5])
data.add_data([1,0.5],[1.5])
data.add_data([0,0.5],[0.5])
data.add_data([0,0.5],[0.5])
data.add_data([1,1.5],[2.5])
data.add_data([1.5,1],[2.5])
data.add_data([1,1],[2])
data.add_data([-1,-1],[-2])
data.add_data([-1,-0.5],[-1.5])
data.add_data([-1,2],[1])
data.add_data([-1,3],[2])






# create trainer
# this trains the network in a later step
# train = trainer(net,50,data)

#print network loss before training
# print(train.get_full_data_loss())


# #start training
# train.start_train()

#print network loss after training
# print(train.get_full_data_loss())

# #get the network
# net = train.get_net()

# s = save(net)

# s.save("temp.mnn")

while True:
  sex = float(input("input 1?  "))
  age = float(input("input 2?  "))
  print(net.run([sex,age]))

#print the network's output from all the data
# print(net.run_all_data(data))