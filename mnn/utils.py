from random import random


def mult_lists(list1, list2):
  temp = [a * b for a, b in zip(list1, list2)]
  return temp


def mae_loss(y_true, y_pred):
  sum = 0
  for i in range(len(y_true)):
    if y_true[i] != y_pred[i]:
      sum += abs(y_true[i] - y_pred[i])
  return sum / len(y_true)


def list_avg(list):
  return sum(list) / len(list)

def list_sub(list1,list2):
  temp = [a_i - b_i for a_i, b_i in zip(list1, list2)]
  return temp


def plus_minus(a, b):
  t = random()
  if t >= 0.5:
    return a + b
  else:
    return a - b
