"""
adapted from https://github.com/INM-6/beNNch-models/blob/main/hpc_benchmark/hpc_benchmark.py
"""

import nest
import time
import numpy as np
import os
import scipy.special as sp

M_INFO = 10
M_ERROR = 30

def convert_synapse_weight(tau_m, tau_syn, C_m):
    """
    Computes conversion factor for synapse weight from mV to pA

    This function is specific to the leaky integrate-and-fire neuron
    model with alpha-shaped postsynaptic currents.

    """

    # compute time to maximum of V_m after spike input
    # to neuron at rest
    a = tau_m / tau_syn
    b = 1.0 / tau_syn - 1.0 / tau_m
    t_rise = 1.0 / b * (-lambertwm1(-np.exp(-1.0 / a) / a).real - 1.0 / a)

    v_max = np.exp(1.0) / (tau_syn * C_m * b) * (
            (np.exp(-t_rise / tau_m) - np.exp(-t_rise / tau_syn)) /
            b - t_rise * np.exp(-t_rise / tau_syn))
    return 1. / v_max



class NestBenchmarkNetwork():

    params = {
        'num_threads': 1, #{threads_per_task},  # total number of threads per process
        'simtime': 500, #10_000, #{model_time_sim},  # total simulation time in ms
        'presimtime': 50, #{model_time_presim},  # simulation time until reaching equilibrium
        'dt': 0.1,  # simulation step
        'compressed_spikes': False, #{compressed_spikes},  # whether to use spike compression
        'rng_seed': 42, #{rng_seed},  # random number generator seed
        'path_name': '.',  # path where all files will have to be written
        'log_file': 'logfile',  # naming scheme for the log files
        'step_data_keys': {},  # metrics to be recorded at each time step
        'profile_memory': False, # record memory profile
    }

    tau_syn = 0.32582722403722841

    brunel_params = {

        #'Nrec': 1000,  # number of neurons to record spikes from

        'model_params': {  # Set variables for iaf_psc_alpha
            'E_L': 0.0,  # Resting membrane potential(mV)
            'C_m': 250.0,  # Capacity of the membrane(pF)
            'tau_m': 10.0,  # Membrane time constant(ms)
            't_ref': 0.5,  # Duration of refractory period(ms)
            'V_th': 20.0,  # Threshold(mV)
            'V_reset': 0.0,  # Reset Potential(mV)
            # time const. postsynaptic excitatory currents(ms)
            'tau_syn_ex': tau_syn,
            # time const. postsynaptic inhibitory currents(ms)
            'tau_syn_in': tau_syn,
            #'tau_minus': 30.0,  # time constant for STDP(depression)
            # V can be randomly initialized see below
            'V_m': 5.7  # mean value of membrane potential
        },

        ####################################################################
        # Note that Kunkel et al. (2014) report different values. The values
        # in the paper were used for the benchmarks on K, the values given
        # here were used for the benchmark on JUQUEEN.

        'randomize_Vm': True,
        'mean_potential': 5.7,
        'sigma_potential': 7.2,

        # synaptic weight
        'JE': 0.14,  # peak of EPSP

        'sigma_w': 3.47,  # standard dev. of E->E synapses(pA)
        'g': -5.0,

        'eta': 1.685,  # scaling of external stimulus
        'filestem': params['path_name']
    }


    def __init__(self, scale, CE, CI, weight_excitatory, weight_inhibitory, delay, nrec, extra_kernel_params=None):
        nest.ResetKernel()
        nest.set_verbosity(M_INFO)

        self.NE = int(9000 * scale)
        self.NI = int(2250 * scale)
        self.CE = int(CE)
        self.CI = int(CI)

        self.weight_excitatory = weight_excitatory
        self.weight_inhibitory = weight_inhibitory

        self.delay = delay

        self.nrec = min(nrec, self.NE)

        # set global kernel parameters
        nest.SetKernelStatus({'local_num_threads': self.params['num_threads'],
                              'resolution': self.params['dt'],
                              'rng_seed': self.params['rng_seed'],
                              'overwrite_files': True,
                              'use_compressed_spikes': self.params['compressed_spikes'],
                              'keep_source_table': False})
        if extra_kernel_params:
            nest.SetKernelStatus(extra_kernel_params)










    def build_network(self):
        """Builds the network including setting of simulation and neuron
        parameters, creation of neurons and connections
        """
        tic = time.time()  # start timer on construction

        model_params = self.brunel_params['model_params']

        nest.message(M_INFO, 'build_network', 'Creating excitatory population.')
        E_neurons = nest.Create('iaf_psc_exp', self.NE, params=model_params)

        nest.message(M_INFO, 'build_network', 'Creating inhibitory population.')
        I_neurons = nest.Create('iaf_psc_exp', self.NI, params=model_params)

        if self.brunel_params['randomize_Vm']:
            nest.message(M_INFO, 'build_network',
                         'Randomzing membrane potentials.')

            random_vm = nest.random.normal(self.brunel_params['mean_potential'],
                                           self.brunel_params['sigma_potential'])
            nest.GetLocalNodeCollection(E_neurons).V_m = random_vm
            nest.GetLocalNodeCollection(I_neurons).V_m = random_vm



        nest.message(M_INFO, 'build_network',
                     'Creating excitatory stimulus generator.')

        # excitatory stimulus
        # Convert synapse weight from mV to pA
        conversion_factor = convert_synapse_weight(
        model_params['tau_m'], model_params['tau_syn_ex'], model_params['C_m'])
        JE_pA = conversion_factor * self.brunel_params['JE']

        nu_thresh = model_params['V_th'] / (
                self.CE * model_params['tau_m'] / model_params['C_m'] *
                JE_pA * np.exp(1.) * self.tau_syn)
        nu_ext = nu_thresh * self.brunel_params['eta']

        # E_stimulus = nest.Create('poisson_generator', 1, {
        #     'rate': nu_ext * CE_expected * 1000.})

        E_stimulus = nest.Create('dc_generator', 1, {'amplitude': 100.})


        # spike recorder
        nest.message(M_INFO, 'build_network',
                     'Creating excitatory spike recorder.')

        recorder_label = os.path.join(
            self.brunel_params['filestem'],
            #'alpha_' + str(stdp_params['alpha']) + '_spikes')
            'spikes')
        E_recorder = nest.Create('spike_recorder', params={
            'record_to': 'ascii',
            'label': recorder_label
        })
        # TODO save spieks to .dat or just print something?

        BuildNodeTime = time.time() - tic
        node_memory = str(get_vmsize())
        node_memory_rss = str(get_rss())
        node_memory_peak = str(get_vmpeak())

        tic = time.time()



        self.connect_neurons(E_neurons, I_neurons, self.CE, self.CI, E_stimulus)




        if self.params['num_threads'] != 1:
            local_neurons = nest.GetLocalNodeCollection(E_neurons)
            # GetLocalNodeCollection returns a stepped composite NodeCollection, which
            # cannot be sliced. In order to allow slicing it later on, we're creating a
            # new regular NodeCollection from the plain node IDs.
            local_neurons = nest.NodeCollection(local_neurons.tolist())
        else:
            local_neurons = E_neurons

        if len(local_neurons) < self.nrec:
            nest.message(
                M_ERROR, 'build_network',
                """Spikes can only be recorded from local neurons, but the
                number of local neurons is smaller than the number of neurons
                spikes should be recorded from. Aborting the simulation!""")
            exit(1)

        nest.message(M_INFO, 'build_network', 'Connecting spike recorders.')
        nest.Connect(local_neurons[:self.nrec], E_recorder,
                     'all_to_all', 'static_synapse_hpc')
        #nest.Connect(local_neurons, E_recorder,
        #             'all_to_all', 'static_synapse_hpc')

        # read out time used for building
        BuildEdgeTime = time.time() - tic
        network_memory = str(get_vmsize())
        network_memory_rss = str(get_rss())
        network_memory_peak = str(get_vmpeak())

        d = {'py_time_create': BuildNodeTime,
             'py_time_connect': BuildEdgeTime,
             'node_memory': node_memory,
             'node_memory_rss': node_memory_rss,
             'node_memory_peak': node_memory_peak,
             'network_memory': network_memory,
             'network_memory_rss': network_memory_rss,
             'network_memory_peak': network_memory_peak}
        recorders = E_recorder

        return d, recorders




    def connect_neurons(self, E_neurons, I_neurons, CE, CI, E_stimulus):

        nest.SetDefaults('static_synapse_hpc', {'delay': self.delay})
        nest.CopyModel('static_synapse_hpc', 'syn_ex',
                       {'weight': self.weight_excitatory})
        nest.CopyModel('static_synapse_hpc', 'syn_in',
                       {'weight': self.weight_inhibitory})

        # Connect dc generator to neurons
        nest.message(M_INFO, 'build_network', 'Connecting stimulus generators.')
        nest.Connect(E_stimulus, E_neurons, {'rule': 'all_to_all'},
                     {'synapse_model': 'syn_ex'})
        nest.Connect(E_stimulus, I_neurons, {'rule': 'all_to_all'},
                     {'synapse_model': 'syn_ex'})


        # excitatory -> excitatory
        nest.message(M_INFO, 'build_network',
                     'Connecting excitatory -> excitatory population.')
        nest.Connect(E_neurons, E_neurons,
                     {'rule': 'fixed_indegree', 'indegree': CE,
                      'allow_autapses': False, 'allow_multapses': True},
                     {'synapse_model': 'syn_ex'})

        # inhibitory -> excitatory
        nest.message(M_INFO, 'build_network',
                     'Connecting inhibitory -> excitatory population.')
        nest.Connect(I_neurons, E_neurons,
                     {'rule': 'fixed_indegree', 'indegree': CI,
                      'allow_autapses': False, 'allow_multapses': True},
                     {'synapse_model': 'syn_in'})

        # excitatory -> inhibitory
        nest.message(M_INFO, 'build_network',
                     'Connecting excitatory -> inhibitory population.')
        nest.Connect(E_neurons, I_neurons,
                     {'rule': 'fixed_indegree', 'indegree': CE,
                      'allow_autapses': False, 'allow_multapses': True},
                     {'synapse_model': 'syn_ex'})

        # inhibitry -> inhibitory
        nest.message(M_INFO, 'build_network',
                     'Connecting inhibitory -> inhibitory population.')
        nest.Connect(I_neurons, I_neurons,
                     {'rule': 'fixed_indegree', 'indegree': CI,
                      'allow_autapses': False, 'allow_multapses': True},
                     {'synapse_model': 'syn_in'})










    def run_simulation(self):
        """Performs a simulation, including network construction"""


        if self.params['profile_memory']:
            base_memory = str(get_vmsize())
            base_memory_rss = str(get_rss())
            base_memory_peak = str(get_vmpeak())

            build_dict, sr = self.build_network()

            tic = time.time()

            nest.Prepare()

            InitTime = time.time() - tic
            init_memory = str(get_vmsize())
            init_memory_rss = str(get_rss())
            init_memory_peak = str(get_vmpeak())

            presim_steps = int(self.params['presimtime'] // nest.min_delay)
            presim_remaining_time = self.params['presimtime'] - (presim_steps * nest.min_delay)
            sim_steps = int(self.params['simtime'] // nest.min_delay)
            sim_remaining_time = self.params['simtime'] - (sim_steps * nest.min_delay)

            total_steps = presim_steps + sim_steps + (1 if presim_remaining_time > 0 else 0) + (
                1 if sim_remaining_time > 0 else 0)
            times, vmsizes, vmpeaks, vmrsss = (
            np.empty(total_steps), np.empty(total_steps), np.empty(total_steps), np.empty(total_steps))
            step_data = {key: np.empty(total_steps) for key in self.params['step_data_keys']}
            tic = time.time()

            for d in range(presim_steps):
                nest.Run(nest.min_delay)
                times[d] = time.time() - tic
                vmsizes[presim_steps] = get_vmsize()
                vmpeaks[presim_steps] = get_vmpeak()
                vmrsss[presim_steps] = get_rss()
                for key in self.params['step_data_keys']:
                    step_data[key][d] = getattr(nest, key)

            if presim_remaining_time > 0:
                nest.Run(presim_remaining_time)
                times[presim_steps] = time.time() - tic
                vmsizes[presim_steps + sim_steps] = get_vmsize()
                vmpeaks[presim_steps + sim_steps] = get_vmpeak()
                vmrsss[presim_steps + sim_steps] = get_rss()
                for key in self.params['step_data_keys']:
                    step_data[key][presim_steps] = getattr(nest, key)
                presim_steps += 1

            PreparationTime = time.time() - tic

            intermediate_kernel_status = nest.kernel_status

            tic = time.time()

            for d in range(sim_steps):
                nest.Run(nest.min_delay)
                times[presim_steps + d] = time.time() - tic
                for key in self.params['step_data_keys']:
                    step_data[key][presim_steps + d] = getattr(nest, key)

            if sim_remaining_time > 0:
                nest.Run(sim_remaining_time)
                times[presim_steps + sim_steps] = time.time() - tic
                for key in self.params['step_data_keys']:
                    step_data[key][presim_steps + sim_steps] = getattr(nest, key)
                sim_steps += 1

            SimCPUTime = time.time() - tic
            total_memory = str(get_vmsize())
            total_memory_rss = str(get_rss())
            total_memory_peak = str(get_vmpeak())



        if not self.params['profile_memory']:
            build_dict, sr = self.build_network()

            tic = time.time()
            base_memory = str(get_vmsize())
            nest.Prepare()

            InitTime = time.time() - tic
            init_memory = str(get_vmsize())

            tic = time.time()
            nest.Run(self.params['presimtime'])
            PreparationTime = time.time() - tic

            intermediate_kernel_status = nest.kernel_status

            tic = time.time()
            nest.Run(self.params['simtime'])
            SimCPUTime = time.time() - tic
            total_memory = str(get_vmsize())

        average_rate = self.compute_rate(sr)

        d = {'py_time_init': InitTime,
             'py_time_presimulate': PreparationTime,
             'py_time_simulate': SimCPUTime,
             'average_rate': average_rate,
             'base_memory': base_memory,
             'init_memory': init_memory,
             'total_memory': total_memory}

        if self.params['profile_memory']:
            memory_dict = {'base_memory_rss': base_memory_rss,
                           'init_memory_rss': init_memory_rss,
                           'total_memory_rss': total_memory_rss,
                           'base_memory_peak': base_memory_peak,
                           'init_memory_peak': init_memory_peak,
                           'total_memory_peak': total_memory_peak}

            d.update(memory_dict)

        d.update(build_dict)
        final_kernel_status = nest.kernel_status
        d.update(final_kernel_status)

        # Subtract timer information from presimulation period
        timers = ['time_collocate_spike_data', 'time_communicate_prepare',
                  'time_communicate_spike_data', 'time_deliver_spike_data',
                  'time_gather_spike_data', 'time_update', 'time_simulate']

        for timer in timers:
            try:
                d[timer + '_presim'] = intermediate_kernel_status[timer]
                d[timer] -= intermediate_kernel_status[timer]
            except KeyError:
                # KeyError if compiled without detailed timers, except time_simulate
                continue
        print(d)

        nest.Cleanup()

        if self.params['profile_memory']:
            fn = '{fn}_{rank}_steps.dat'.format(fn=self.params['log_file'], rank=nest.Rank())
            with open(fn, 'w') as f:
                f.write('time ' + ' '.join(self.params['step_data_keys']) + '\n')
                for d in range(presim_steps + sim_steps):
                    f.write(str(times[d]) + ' ' + ' '.join(str(step_data[key][d]) for key in self.params['step_data_keys']) + '\n')

        return average_rate



    def compute_rate(self, sr):
        """Compute local approximation of average firing rate

        This approximation is based on the number of local nodes, number
        of local spikes and total time. Since this also considers devices,
        the actual firing rate is usually underestimated.

        """

        n_local_spikes = sr.n_events
        n_local_neurons = self.nrec
        simtime = self.params['simtime']
        return 1. * n_local_spikes / (n_local_neurons * simtime) * 1e3

















def _VmB(VmKey):
    _proc_status = '/proc/%d/status' % os.getpid()
    _scale = {'kB': 1024.0, 'mB': 1024.0 * 1024.0, 'KB': 1024.0, 'MB': 1024.0 * 1024.0}
    # get pseudo file  /proc/<pid>/status
    try:
        t = open(_proc_status)
        v = t.read()
        t.close()
    except:
        return 0.0  # non-Linux?
    # get VmKey line e.g. 'VmRSS:  9999  kB\n ...'
    i = v.index(VmKey)
    v = v[i:].split(None, 3)  # whitespace
    if len(v) < 3:
        return 0.0  # invalid format?
    # convert Vm value to bytes
    return float(v[1]) * _scale[v[2]]


def get_vmsize(since=0.0):
    """Return memory usage in bytes."""
    return _VmB('VmSize:') - since


def get_rss(since=0.0):
    """Return resident memory usage in bytes."""
    return _VmB('VmRSS:') - since


def get_vmpeak(since=0.0):
    """Return peak memory usage in bytes."""
    return _VmB('VmPeak:') - since


def lambertwm1(x):
    """Wrapper for LambertWm1 function"""
    # Using scipy to mimic the gsl_sf_lambert_Wm1 function.
    return sp.lambertw(x, k=-1 if x < 0 else 0).real
