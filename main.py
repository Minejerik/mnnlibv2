from mnn import network, layer, activations as a, dataset, trainer


#create network
net = network()
#create input layer w/ relu activation function
#it has 2 input and 4 output nodes
lay = layer(2,4,a.relu)
net.add_layer(lay)
#create hidden layer also w/ relu activation function
#this time w/ 4 input nodes and 4 output nodes
lay = layer(4,4,a.relu)
net.add_layer(lay)
#create output layer w/ straight activation function
#it has 4 input nodes and 1 output node
lay = layer(4,1,a.straight)
net.add_layer(lay)

#print the network
#allows for easy viewing
print(net)

#create dataset
data = dataset()


#add some data to it
#this emulates an and gate
data.add_data([0,0],[0])
data.add_data([0,1],[1])
data.add_data([1,0],[1])
data.add_data([1,1],[1])


#create trainer
#this trains the network in a later step
train = trainer(net,60,data)

#print network loss before training
print(train.get_full_data_loss())

#start training
train.start_train()

#print network loss after training
print(train.get_full_data_loss())

#get the network
net = train.get_net()


#print the network's output from all the data
print(net.run_all_data(data))