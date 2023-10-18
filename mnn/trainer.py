from mnn.utils import mae_loss, list_avg
from random import choice, random

class trainer:
  def __init__(self, net, epochs, data,learn_rate=0.0000001):
    self.net = net
    self.epochs = epochs
    self.data = data
    self.learn_rate = learn_rate
    self.debug = False

  def get_full_data_loss(self):
    losses = []
    for i,o in zip(self.data.inps, self.data.outs):
      losses.append(mae_loss(self.net.run(i),o))
    return list_avg(losses)

  def get_net(self):
    return self.net

  def print(self,string):
    if self.debug == True:
      print(string)

  def start_train(self):
    for epoch in range(self.epochs):
      self.print(f"Starting epoch {epoch}")
      
      old_loss = self.get_full_data_loss()
      weights = self.net.get_weights()
      
      self.print(f"Starting Loss: {old_loss}")
      
      temp_weight = choice(weights)
      old_weight = temp_weight.weight
      temp_weight.weight = random()
      new_loss = self.get_full_data_loss()
      
      self.print(f"New Loss: {new_loss}")

      if new_loss < old_loss:
        self.print("hell yeah")
      else:
        temp_weight.weight = old_weight
        self.print("nope")
      
      
    