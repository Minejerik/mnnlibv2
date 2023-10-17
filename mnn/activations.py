import math

def relu(inp):
  if inp < 0:
    return 0
  else:
    return inp

def sigmoid(inp):
  return 1/(1+math.exp(-inp))

def straight(inp):
  return inp
