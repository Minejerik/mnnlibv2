from mnn.utils import mult_lists
from mnn.weight import weight
from random import uniform

class neuron():
  def __init__(self,inp_count,act_func):
    self.weights = []
    self.activation = act_func
    for _ in range(inp_count):
      self.weights.append(weight(uniform(-1.0,1.0)))
    self.inp_count = inp_count

  def run(self, input):
    w_l = [w.get_weight() for w in self.weights]
    temp = mult_lists(input,w_l)
    temp = sum(temp)
    temp = self.activation(inp=temp)
    return temp