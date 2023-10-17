from mnn.utils import mult_lists
from random import uniform

class neuron():
  def __init__(self,inp_count,act_func):
    self.weights = []
    self.activation = act_func
    for _ in range(inp_count):
      self.weights.append(uniform(-1.0,1.0))
    self.inp_count = inp_count

  def run(self, input):
    temp = mult_lists(input,self.weights)
    temp = sum(temp)
    temp = self.activation(inp=temp)
    return temp