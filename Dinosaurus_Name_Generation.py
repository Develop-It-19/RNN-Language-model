#Import Dependencies
import numpy as np
from utils import *
import random

#Dataset and Preprocessing
data = open('dinos.txt', 'r').read()
data = data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i, ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i, ch in enumerate(sorted(chars)) }
print(ix_to_char)

#Gradient Clipping: to avoid exploding gradients.
def clip(gradients, maxValue):
  dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']
  
  for gradient in [dWaa, dWax, dWya, db, dby]:
    np.clip(gradient, -maxValue, maxValue, out = gradient)
  
  gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
  
  return gradients

np.random.seed(3)
  #We Multiply with 10 so that we get large numbers to show exploding gradients.
dWax = np.random.randn(5, 3) * 10
dWaa = np.random.randn(5, 5) * 10
dWya = np.random.randn(2, 5) * 10
db = np.random.randn(5, 1) * 10
dby = np.random.randn(2, 1) * 10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])

#Sampling: a technique used to generate characters
def sample(parameters, char_to_ix, seed):
  Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
  vocab_size = by.shape[0]
  n_a = Waa.shape[1]
  
  x = np.zeros((vocab_size, 1))
  a_prev = np.zeros((n_a, 1))
  
  indices = []
  
  idx = -1  #A Flag initialized for new line character to determine the end of sentence.
  
  counter = 0   #We will stop if we reach 50 characters. To prevent infinite loop.
  newline_character = char_to_ix['\n']
  
  while(idx != newline_character and counter != 50):
    a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
    z = np.dot(Wya, a) + by
    y = softmax(z)
    
    np.random.seed(counter + seed)
    
    idx = np.random.choice(list(range(vocab_size)), p = y.ravel())
    
    indices.append(idx)
    
    x = np.zeros((vocab_size, 1))
    x[idx] = 1
    
    a_prev = a
    
    seed += 1
    counter += 1
    
  if(counter == 50):
    indices.append(char_to_ix['\n'])
  
  return indices

np.random.seed(2)
_, n_a = 20, 100







