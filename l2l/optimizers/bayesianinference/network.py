"""
Definition of the NEST network
"""

import nest
import numpy as np

class Network:
    def __init__(self, w_ex, w_in, delay, p_ex, p_in):
        # "fixed" parameters
        self.scale = 2500
        self.N_ex = 4*self.scale # 80% excitatory neurons
        self.N_in = 1*self.scale # 20% inhibitory neurons
        self.N_rec = 50 # number of neurons that are recorded
        self.poisson_rate = self.scale*50/4 # 10-100 per neuron
        self.wait_time = 100
        self.check_time = 100
        self.theta_labels = ['w_ex', 'w_in', 'delay', 'p_ex', 'p_in']
        
        # "free" oarameters
        self.w_ex = w_ex
        self.w_in = w_in
        self.delay = delay
        self.p_ex = p_ex
        self.p_in = p_in
        
    def create_nodes(self):
        # neurons
        self.nodes_ex = nest.Create("iaf_psc_alpha", self.N_ex)
        self.nodes_in = nest.Create("iaf_psc_alpha", self.N_in)
    
        # Poisson input (noise)
        self.noise = nest.Create("poisson_generator", params={"rate": self.poisson_rate})
    
        # recorders
        self.recorder_ex = nest.Create("spike_recorder", params={'record_to': 'memory', 'start': self.wait_time})
        self.recorder_in = nest.Create("spike_recorder", params={'record_to': 'memory', 'start': self.wait_time})
    
        return self.nodes_ex, self.nodes_in, self.noise, self.recorder_ex, self.recorder_in
    
    def connect_nodes(self):
        nest.CopyModel("static_synapse", "excitatory", {"weight": self.w_ex, "delay": self.delay})
        nest.CopyModel("static_synapse", "inhibitory", {"weight": self.w_in, "delay": self.delay})
    
        # from Poisson to neurons
        nest.Connect(self.noise, self.nodes_ex, syn_spec="excitatory")
        nest.Connect(self.noise, self.nodes_in, syn_spec="excitatory")
    
        # from neurons to recorders
        nest.Connect(self.nodes_ex[:self.N_rec], self.recorder_ex, syn_spec="excitatory")
        nest.Connect(self.nodes_in[:self.N_rec], self.recorder_in, syn_spec="excitatory")
    
        # excitatory neurons
        conn_params_ex = {'rule': 'pairwise_bernoulli', 'p': self.p_ex}
        nest.Connect(self.nodes_ex, self.nodes_ex + self.nodes_in, conn_params_ex, "excitatory")
    
        # inhibitory neurons
        conn_params_in = {'rule': 'pairwise_bernoulli', 'p': self.p_in}
        nest.Connect(self.nodes_in, self.nodes_ex + self.nodes_in, conn_params_in, "inhibitory")
        
    def calculate_rates(self, simtime):
        # calculate firing rates
        n_events_ex = self.recorder_ex.n_events
        n_events_in = self.recorder_in.n_events
        rate_ex = n_events_ex / simtime * 1000.0 / self.N_rec
        rate_in = n_events_in / simtime * 1000.0 / self.N_rec
        return rate_ex, rate_in
    
    def simulate_network(self, simtime):
        # simulate
        nest.Prepare()
        nest.Run(self.wait_time+self.check_time) # record 100 ms
    
        # calculate firing rates
        rate_ex, rate_in = self.calculate_rates(simtime=self.check_time)
    
        # check if rate is too high
        if rate_ex > 200 or rate_in > 200:
            return np.nan, np.nan
        
        nest.Run(self.simtime-self.check_time) # run for the rest of the simtime
        nest.Cleanup()
        
        # calculate firing rates
        rate_ex, rate_in = self.calculate_rates(simtime=self.simtime)
    
        return rate_ex, rate_in
