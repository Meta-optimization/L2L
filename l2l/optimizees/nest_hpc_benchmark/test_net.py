from network import NestBenchmarkNetwork
import random

scale = 0.1

#individual = {  'weight_ex':  random.uniform(1     , 20),
#                'weight_in':  random.uniform(-100  , -5),
#                'pCE':        random.uniform(0     , 1),
#                'pCI':        random.uniform(0     , 1),
#                'delay':      random.uniform(0.1   , 10),
#                }  
  
individual = {  'weight_ex':  random.uniform(0     , 200),
                'weight_in':  random.uniform(-1000  , 0),
                'pCE':        random.uniform(0     , 1),
                'pCI':        random.uniform(0     , 1),
                'delay':      random.uniform(0.1   , 10),
                }   


weight_ex = individual['weight_ex']
weight_in = individual['weight_in']
pCE = individual['pCE']
pCI = individual['pCI']
delay = individual['delay']

net = NestBenchmarkNetwork(scale, pCE, pCI, weight_ex, weight_in, delay=delay)
average_rate = net.run_simulation()

print("")
print("average firing rate:", average_rate)