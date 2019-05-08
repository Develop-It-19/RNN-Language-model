#Basic RNN Cell.
#One Layer.
#Python

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
Wax, Waa, Wya, b, by = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a), np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}

indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
  loss, cache = rnn_forward(X, Y, a_prev, parameters)
  gradients, a = rnn_backward(X, Y, parameters, cache)
  gradients = clip(gradients, 5)
  parameters = update_parameters(parameters, gradients, learning_rate)
  
  return loss, gradients, a[len(X) - 1]

np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya, b, by = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a), np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])

#Training the Model
def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
  n_x, n_y = vocab_size, vocab_size
  parameters = initialize_parameters(n_a, n_x, n_y)
  loss = get_initial_loss(vocab_size, dino_names)   #To Smooth the loss
  
  with open("dinos.txt") as f:
    examples = f.readlines()
  examples = [x.lower().strip() for x in examples]
  
  np.random.seed(0)
  np.random.shuffle(examples)
  
  a_prev = np.zeros((n_a, 1))
  
  for j in range(num_iterations):
    index = j % len(examples)
    X = [None] + [char_to_ix[ch] for ch in examples[index]]
    Y = X[1:] + [char_to_ix["\n"]]
    
    curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
    
    loss = smooth(loss, curr_loss)  #Latency trick to keep the loss smooth in order to accelerate training.
    
    if j % 2000 == 0:
      print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
      
      seed = 0
      for name in range(dino_names):
        sampled_indices = sample(parameters, char_to_ix, seed)
        print_sample(sampled_indices, ix_to_char)
        
        seed += 1
      print('\n')
  return parameters

parameters = model(data, ix_to_char, char_to_ix)


    




