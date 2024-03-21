import math

#just activation functions
#they are passed as a paramater to the layers

def relu(inp:float):
  """
  Basic rectified linear unit (relu) activation function
  works for most basic networks
  if x < 0 : return 0
  else return x
  """
  return 0 if inp < 0 else inp


def sigmoid(inp:float):
  """
  Sigmoid activation function
  good for having negative numbers
  and for more complex networks
  """
  return 1 / (1 + math.exp(-inp))


def straight(inp:float):

  """
  Straight activation function
  very basic
  just returns the input
  good for simple networks
  """
  
  return inp


def binary(inp:float):
  """
  binary activation function
  if x > 0 : return 1
  else return 0
  good for binary classification
  """
  
  return 1 if inp > 0 else 0


def leakyrelu(inp:float):
  """
  Leaky relu activation function
  in most cases better than relu
  """

  return inp if inp > 0 else inp * 0.01
