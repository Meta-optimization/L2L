import argparse
import json
from mpi4py import MPI
import nest
import numpy as np
import os
import pandas as pd
import pathlib
import pickle
import time

from collections import OrderedDict
from itertools import permutations, product
from l2l.optimizees.snn import spike_generator, visualize


class Reservoir:
    """
    Class reservoir network (LSM) with LIF neurons works with NEST 3
    """
    def __init__(self):
        fp = pathlib.Path(__file__).parent.absolute()
        print("Using config file: ", os.path.join(str(fp), "config.json"))
        with open(os.path.join(str(fp), "config.json")) as jsonfile:
            self.config = json.load(jsonfile)
        self.t_sim = self.config["t_sim"]
        self.input_type = self.config["input_type"]
        # Resolution, simulation steps in [ms]
        self.dt = self.config["dt"]
        self.neuron_model = self.config["neuron_model"]

        # Number of neurons per layer
        self.n_input_neurons = self.config["n_input"]
        self.n_bulk_ex_neurons = self.config["n_bulk_ex"]
        self.n_bulk_in_neurons = self.config["n_bulk_in"]
        self.n_neurons_out_e = self.config["n_out_ex"]
        self.n_neurons_out_i = self.config["n_out_in"]
        self.n_output_clusters = self.config["n_output_clusters"]
        self.psc_e = self.config["psc_e"]
        self.psc_i = self.config["psc_i"]
        self.psc_ext = self.config["psc_ext"]
        self.bg_rate = self.config["bg_rate"]
        self.record_interval = self.config["record_interval"]
        self.warm_up_time = self.config["warm_up_time"]
        self.cooling_time = self.config["cooling_time"]

        # Init of nodes
        self.nodes_in = None
        self.nodes_e = None
        self.nodes_i = None
        self.nodes_out_e = []
        self.nodes_out_i = []
        # Init of generators and noise
        self.pixel_rate_generators = None
        self.noise = None
        # Init of spike detectors
        self.input_spike_detector = None
        self.bulks_detector_ex = None
        self.bulks_detector_in = None
        self.out_detector_e = None
        self.out_detector_i = None
        self.rates = None
        self.target_px = None
        # Lists for connections
        self.total_connections_e = []
        self.total_connections_i = []
        self.total_connections_out_e = []
        self.total_connections_out_i = []
        # Lists for firing rates
        self.mean_ca_e = []
        self.mean_ca_i = []
        self.mean_ca_out_e = [[] for _ in range(self.n_output_clusters)]
        self.mean_ca_out_i = [[] for _ in range(self.n_output_clusters)]

    def connect_network(self, path="."):
        self.prepare_network()
        # Do the connections
        self.connect_internal_bulk()
        self.connect_external_input()
        self.connect_spike_detectors()
        self.connect_noise_bulk()
        self.connect_internal_out()
        self.connect_bulk_to_out()
        self.connect_noise_out()
        # self.connect_out_to_out()

        # if path does not exist create
        if not os.path.exists(path):
            print(f"Creating path {path}")
            os.mkdir(path)

        # create connection types of
        # {'ee', 'ei', 'eoeo', 'eoio', 'ie', 'ii', 'ioeo', 'ioio'}
        # e/i is the connection type, o indicates output
        # eo means excitatory output connections
        perms = set(["".join(p) for p in permutations("eeii", 2)])
        out_perms = set(["".join(p) for p in product(["eo", "io"], repeat=2)])
        # only eeo, eio, ieo and iio are going to be optimized though
        # set([''.join(p) product(['e', 'i'], ['eo', 'io'], repeat=1)])
        optim_perms = {"eeo", "eio", "ieo", "iio"}
        perms = perms.union(out_perms, optim_perms)
        # need dictionary to store the length specific to the outputs
        d = {}
        for p in perms:
            source, target = self._get_net_structure(p)
            length_conns = 0
            if not isinstance(source, list):
                source = [source]
            if not isinstance(target, list):
                target = [target]
            conns = []
            for s in source:
                for t in target:
                    synapse_collection = nest.GetConnections(source=s, target=t)
                    conns.append(synapse_collection)
                    assert len(synapse_collection.target) == len(synapse_collection.source)
                    length_conns += len(synapse_collection.source)
            # check if file already exists and if not save
            if not os.path.isfile(os.path.join(path, f"{p}_connections.csv")):
                # os.remove(os.path.join(path, f"{p}_connections.csv"))
                self.save_connections(conns, path=path, typ=p)
            d[p] = length_conns
        return d["eeo"], d["eio"], d["ieo"], d["iio"]

    def prepare_network(self):
        """Helper functions to create the network"""
        self.reset_kernel()
        self.create_synapses()
        self.create_nodes()
        self.create_input_spike_detectors()
        self.pixel_rate_generators = self.create_pixel_rate_generator(self.input_type)
        self.noise = nest.Create("poisson_generator")
        nest.PrintNodes()

    def _get_net_structure(self, typ):
        # types: {'ee', 'ei', 'eoeo', 'eoio', 'ie', 'ii', 'ioeo', 'ioio'}
        if typ == "ee":
            return self.nodes_e, self.nodes_e
        elif typ == "ei":
            return self.nodes_e, self.nodes_i
        elif typ == "ie":
            return self.nodes_i, self.nodes_e
        elif typ == "ii":
            return self.nodes_i, self.nodes_i
        elif typ == "eeo":
            return self.nodes_e, self.nodes_out_e
        elif typ == "eio":
            return self.nodes_e, self.nodes_out_i
        elif typ == "ieo":
            return self.nodes_i, self.nodes_out_e
        elif typ == "iio":
            return self.nodes_i, self.nodes_out_i
        elif typ == "eoeo":
            return self.nodes_out_e, self.nodes_out_e
        elif typ == "eoio":
            return self.nodes_out_e, self.nodes_out_i
        elif typ == "ioeo":
            return self.nodes_out_i, self.nodes_out_e
        elif typ == "ioio":
            return self.nodes_out_i, self.nodes_out_i

    def reset_kernel(self):
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")
        nest.local_num_threads = int(self.config['threads'])
        nest.resolution = self.dt
        nest.rng_seed = int(self.config["seed"])
        nest.overwrite_files = True

    def create_nodes(self):
        self.nodes_in = nest.Create(self.neuron_model, self.n_input_neurons)
        self.nodes_e = nest.Create(self.neuron_model, self.n_bulk_ex_neurons)
        self.nodes_i = nest.Create(self.neuron_model, self.n_bulk_in_neurons)
        for i in range(self.n_output_clusters):
            self.nodes_out_e.append(
                nest.Create(self.neuron_model, self.n_neurons_out_e)
            )
            self.nodes_out_i.append(
                nest.Create(self.neuron_model, self.n_neurons_out_i)
            )

    def create_input_spike_detectors(self, record_fr=True):
        self.input_spike_detector = nest.Create("spike_recorder")
        if record_fr:
            # if spikes should be recorder to disk use keyword 'record_to'
            self.bulks_detector_ex = nest.Create(
                "spike_recorder", params={"label": "bulk_ex"}
            )
            self.bulks_detector_in = nest.Create(
                "spike_recorder", params={"label": "bulk_in"}
            )
            self.out_detector_e = nest.Create(
                "spike_recorder", self.n_output_clusters, params={"label": "out_e"}
            )
            self.out_detector_i = nest.Create(
                "spike_recorder", self.n_output_clusters, params={"label": "out_i"}
            )

    def create_pixel_rate_generator(self, input_type):
        if input_type == "greyvalue":
            return nest.Create("poisson_generator", self.n_input_neurons)
        elif input_type == "bellec":
            return nest.Create("spike_generator", self.n_input_neurons)
        elif input_type == "greyvalue_sequential":
            n_img = self.n_input_neurons
            rates, starts, ends = spike_generator.greyvalue_sequential(
                self.target_px[n_img],
                start_time=0,
                end_time=783,
                min_rate=0,
                max_rate=10,
            )
            self.rates = rates
            self.pixel_rate_generators = nest.Create("poisson_generator", len(rates))

    @staticmethod
    def create_synapses():
        nest.CopyModel("static_synapse", "random_synapse")
        nest.CopyModel("static_synapse", "random_synapse_i")

    def connect_spike_detectors(self):
        # Input
        nest.Connect(self.nodes_in, self.input_spike_detector)
        # BULK
        nest.Connect(self.nodes_e, self.bulks_detector_ex)
        nest.Connect(self.nodes_i, self.bulks_detector_in)
        # Out
        for j in range(self.n_output_clusters):
            nest.Connect(self.nodes_out_e[j], self.out_detector_e[j])
            nest.Connect(self.nodes_out_i[j], self.out_detector_i[j])

    def connect_noise_bulk(self):
        poisson_gen = nest.Create("poisson_generator", 1, {'rate': 10000.0}, )
        syn_dict = {"synapse_model": "static_synapse", "weight": 1}
        syn_dict_i = {"synapse_model": "static_synapse", "weight": 1}
        nest.Connect(poisson_gen, self.nodes_e, "all_to_all",
                     syn_spec=syn_dict)
        nest.Connect(poisson_gen, self.nodes_i, "all_to_all",
                     syn_spec=syn_dict_i)

    def connect_noise_out(self):
        poisson_gen = nest.Create("poisson_generator", 1, {'rate': 10000.0}, )
        syn_dict = {"synapse_model": "static_synapse", "weight": 1}
        syn_dict_i = {"synapse_model": "static_synapse", "weight": 1}
        for j in range(self.n_output_clusters):
            nest.Connect(
                poisson_gen, self.nodes_out_e[j], "all_to_all", syn_spec=syn_dict
            )
            nest.Connect(
                poisson_gen, self.nodes_out_i[j], "all_to_all", syn_spec=syn_dict_i
            )

    def connect_greyvalue_input(self):
        """Connects input to bulk"""
        indegree = 8
        weight = 400.
        syn_dict_e = {
            "synapse_model": "random_synapse",
            "weight": nest.random.normal(self.psc_e, weight),
        }
        syn_dict_i = {
            "synapse_model": "random_synapse_i",
            "weight": nest.random.normal(self.psc_e, weight),
        }
        syn_dict_input = {
            "synapse_model": "random_synapse",
            "weight": nest.random.normal(self.psc_e, weight),
        }
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one",
                     syn_spec=syn_dict_input)
        # connect input to bulk
        conn_dict = {'rule': 'fixed_indegree', 'indegree': indegree,
                     "allow_autapses": False, "allow_multapses": False
                     }
        nest.Connect(self.nodes_in, self.nodes_e, conn_spec=conn_dict,
                     syn_spec=syn_dict_e)
        nest.Connect(self.nodes_in, self.nodes_i, conn_spec=conn_dict,
                     syn_spec=syn_dict_i)

    def connect_bellec_input(self):
        nest.Connect(self.pixel_rate_generators, self.nodes_in, "one_to_one")
        syn_dict = {
            "synapse_model": "random_synapse",
            "weight": nest.random.uniform(self.psc_i, self.psc_e),
        }
        conn_dict = {
            "rule": "fixed_outdegree",
            "outdegree": int(0.05 * self.n_bulk_ex_neurons),
        }
        nest.Connect(
            self.nodes_in, self.nodes_e, conn_spec=conn_dict, syn_spec=syn_dict
        )
        conn_dict = {
            "rule": "fixed_outdegree",
            "outdegree": int(0.05 * self.n_bulk_in_neurons),
        }
        nest.Connect(
            self.nodes_in, self.nodes_i, conn_spec=conn_dict, syn_spec=syn_dict
        )

    def clear_input(self):
        """
        Sets a very low rate to the input, for the case where no input is
        provided
        """
        generator_stats = [{"rate": 1.0} for _ in range(self.n_input_neurons)]
        nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def connect_internal_bulk(self):
        syn_dict_e = {
            "synapse_model": "random_synapse",
            "weight": nest.random.normal(self.psc_e, 100.0),
        }
        syn_dict_i = {
            "synapse_model": "random_synapse_i",
            "weight": nest.random.normal(self.psc_i, 100.0),
        }
        # Connect bulk
        conn_dict = {
            "rule": "fixed_outdegree",
            "outdegree": int(0.06 * self.n_bulk_ex_neurons),
            "allow_autapses": False,
            "allow_multapses": False,
        }
        nest.Connect(self.nodes_e, self.nodes_e, conn_dict, syn_spec=syn_dict_e)
        conn_dict = {
            "rule": "fixed_outdegree",
            "outdegree": int(0.08 * self.n_bulk_in_neurons),
            "allow_autapses": False,
            "allow_multapses": False,
        }
        nest.Connect(self.nodes_e, self.nodes_i, conn_dict, syn_spec=syn_dict_e)
        conn_dict = {
            "rule": "fixed_outdegree",
            "outdegree": int(0.1 * self.n_bulk_ex_neurons),
            "allow_autapses": False,
            "allow_multapses": False,
        }
        nest.Connect(self.nodes_i, self.nodes_e, conn_dict, syn_spec=syn_dict_i)
        conn_dict = {
            "rule": "fixed_outdegree",
            "outdegree": int(0.08 * self.n_bulk_in_neurons),
            "allow_autapses": False,
            "allow_multapses": False,
        }
        nest.Connect(self.nodes_i, self.nodes_i, conn_dict, syn_spec=syn_dict_i)

    def connect_internal_out(self):
        # Connect out
        conn_dict = {'rule': 'fixed_indegree', 'indegree': 2, "allow_multapses": False}
        syn_dict = {"synapse_model": "random_synapse"}
        conn_dict_i = {'rule': 'fixed_indegree', 'indegree': 2, "allow_multapses": False}
        syn_dict_i = {"synapse_model": "random_synapse_i"}
        for ii in range(self.n_output_clusters):
            nest.Connect(self.nodes_out_e[ii], self.nodes_out_e[ii], conn_dict,
                         syn_spec=syn_dict)
            nest.Connect(self.nodes_out_e[ii], self.nodes_out_i[ii], conn_dict,
                         syn_spec=syn_dict)
            nest.Connect(self.nodes_out_i[ii], self.nodes_out_e[ii],
                         conn_dict_i, syn_spec=syn_dict_i)
            nest.Connect(self.nodes_out_i[ii], self.nodes_out_i[ii],
                         conn_dict_i, syn_spec=syn_dict_i)

    def connect_bulk_to_out(self):
        # Bulk to out
        conn_dict_e = {
            "rule": "fixed_indegree",
            # 0.3 * self.number_out_exc_neurons
            "indegree": int(self.n_bulk_ex_neurons/(self.n_neurons_out_e/4)),
            "allow_multapses": False,
            "allow_autapses": False,
        }
        conn_dict_i = {
            "rule": "fixed_indegree",
            # 0.2 * self.number_out_exc_neurons
            "indegree": int(self.n_bulk_in_neurons/(self.n_neurons_out_e/4)),
            "allow_multapses": False,
            "allow_autapses": False,
        }
        std = 30.0

        syn_dict_e = {
            "synapse_model": "random_synapse",
            "weight": nest.random.normal(self.psc_e, std),
        }
        syn_dict_i = {
            "synapse_model": "random_synapse_i",
            "weight": nest.random.normal(self.psc_i, std),
        }
        for j in range(self.n_output_clusters):
            nest.Connect(self.nodes_e, self.nodes_out_e[j], conn_dict_e, syn_spec=syn_dict_e)
            nest.Connect(self.nodes_e, self.nodes_out_i[j], conn_dict_e, syn_spec=syn_dict_e)
            nest.Connect(self.nodes_i, self.nodes_out_e[j], conn_dict_i, syn_spec=syn_dict_i)
            nest.Connect(self.nodes_i, self.nodes_out_i[j], conn_dict_i, syn_spec=syn_dict_i)

    def connect_out_to_out(self):
        """
        Inhibits the other clusters
        """
        conn_dict = {
            "rule": "all_to_all",
        }
        # 'allow_autapses': False, 'allow_multapses': False}
        syn_dict = {"synapse_model": "static_synapse", "weight": self.psc_i}
        for j in range(self.n_output_clusters):
            for i in range(self.n_output_clusters):
                if j != i:
                    nest.Connect(
                        self.nodes_out_e[j],
                        self.nodes_out_e[i],
                        conn_dict,
                        syn_spec=syn_dict,
                    )

    def connect_external_input(self):
        nest.SetStatus(self.noise, {"rate": self.bg_rate})
        weight = 5.5
        nest.Connect(
            self.noise,
            self.nodes_e,
            "all_to_all",
            {"weight": weight, "delay": 1.0},
        )
        nest.Connect(
            self.noise,
            self.nodes_i,
            "all_to_all",
            {"weight": weight, "delay": 1.0},
        )

        if self.input_type == "bellec":
            self.connect_bellec_input()
        elif self.input_type == "greyvalue":
            self.connect_greyvalue_input()
        # at the moment the connection structure of the sequential input is
        # the same as normal greyvalue input
        elif self.input_type == "greyvalue_sequential":
            self.connect_greyvalue_input()

    def clear_spiking_events(self):
        nest.SetStatus(self.bulks_detector_ex, "n_events", 0)
        nest.SetStatus(self.bulks_detector_in, "n_events", 0)
        for i in range(self.n_output_clusters):
            nest.SetStatus(self.out_detector_e[i], "n_events", 0)
            nest.SetStatus(self.out_detector_i[i], "n_events", 0)

    def record_fr(self, indx, gen_idx, path, record_out=False, save=True):
        """Records firing rates"""
        n_recorded_bulk_ex = self.n_bulk_ex_neurons
        n_recorded_bulk_in = self.n_bulk_in_neurons
        self.mean_ca_e.append(
            nest.GetStatus(self.bulks_detector_ex, "n_events")[0]
            * 1000.0
            / (self.record_interval * n_recorded_bulk_ex)
        )
        self.mean_ca_i.append(
            nest.GetStatus(self.bulks_detector_in, "n_events")[0]
            * 1000.0
            / (self.record_interval * n_recorded_bulk_in)
        )
        if record_out:
            for i in range(self.n_output_clusters):
                self.mean_ca_out_e[i].append(
                    nest.GetStatus(self.out_detector_e[i], "n_events")[0]
                    * 1000.0
                    / (self.record_interval * self.n_neurons_out_e)
                )
                self.mean_ca_out_i[i].append(
                    nest.GetStatus(self.out_detector_i[i], "n_events")[0]
                    * 1000.0
                    / (self.record_interval * self.n_neurons_out_i)
                )
        if gen_idx % 10 == 0:
            spikes = nest.GetStatus(self.bulks_detector_in, keys="events")[0]
            visualize.spike_plot(
                spikes,
                "Bulk spikes in",
                idx=indx,
                gen_idx=gen_idx,
                save=save,
                path=path,
            )
            spikes = nest.GetStatus(self.bulks_detector_ex, keys="events")[0]
            visualize.spike_plot(
                spikes,
                "Bulk spikes ex",
                idx=indx,
                gen_idx=gen_idx,
                save=save,
                path=path,
            )
            spikes_out_e = nest.GetStatus(self.out_detector_e[0], keys="events")
            visualize.spike_plot(
                spikes_out_e[0],
                "Out spikes ex 0",
                idx=indx,
                gen_idx=gen_idx,
                save=save,
                path=path,
            )
            spikes_out_e = nest.GetStatus(self.out_detector_e[1], keys="events")
            visualize.spike_plot(
                spikes_out_e[0],
                "Out spikes ex 1",
                idx=indx,
                gen_idx=gen_idx,
                save=save,
                path=path,
            )
            spikes_out_i = nest.GetStatus(self.out_detector_i[0], keys="events")
            visualize.spike_plot(
                spikes_out_i[0],
                "Out spikes in 0",
                idx=indx,
                gen_idx=gen_idx,
                save=save,
                path=path,
            )
            spikes_out_i = nest.GetStatus(self.out_detector_i[1], keys="events")
            visualize.spike_plot(
                spikes_out_i[0],
                "Out spikes in 1",
                idx=indx,
                gen_idx=gen_idx,
                save=save,
                path=path,
            )

    def record_ca(self, record_out=False):
        ca_e = (nest.GetStatus(self.nodes_e, "Ca"),)  # Calcium concentration
        self.mean_ca_e.append(np.mean(ca_e))
        ca_i = (nest.GetStatus(self.nodes_i, "Ca"),)  # Calcium concentration
        self.mean_ca_i.append(np.mean(ca_i))
        if record_out:
            for ii in range(self.n_output_clusters):
                # Calcium concentration
                ca_e = (nest.GetStatus(self.nodes_out_e[ii], "Ca"),)
                self.mean_ca_out_e[ii].append(np.mean(ca_e))
                ca_i = (nest.GetStatus(self.nodes_out_i[ii], "Ca"),)
                self.mean_ca_out_i[ii].append(np.mean(ca_i))

    def clear_records(self):
        self.mean_ca_i.clear()
        self.mean_ca_e.clear()
        self.mean_ca_out_e.clear()
        self.mean_ca_out_e = [[] for _ in range(self.n_output_clusters)]
        self.mean_ca_out_i.clear()
        self.mean_ca_out_i = [[] for _ in range(self.n_output_clusters)]
        nest.SetStatus(self.input_spike_detector, {"n_events": 0})

    def record_connectivity(self):
        syn_elems_e = nest.GetStatus(self.nodes_e, "synaptic_elements")
        syn_elems_i = nest.GetStatus(self.nodes_i, "synaptic_elements")
        self.total_connections_e.append(
            sum(neuron["Bulk_E_Axn"]["z_connected"] for neuron in syn_elems_e)
        )
        self.total_connections_i.append(
            sum(neuron["Bulk_I_Axn"]["z_connected"] for neuron in syn_elems_i)
        )

    def set_external_input(self, iteration, train_data, target, path, save):
        # Save image for reference
        if save:
            visualize.plot_image(
                image=train_data,
                random_id=target,
                iteration=iteration,
                path=path,
                save=save,
            )
            visualize.plot_image(
                image=train_data,
                random_id=target,
                iteration=iteration,
                path=path,
                save=save,
            )
        if self.input_type == "greyvalue":
            rates = spike_generator.greyvalue(train_data, min_rate=1, max_rate=100)
            generator_stats = [{"rate": w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        elif self.input_type == "greyvalue_sequential":
            rates = spike_generator.greyvalue_sequential(
                train_data, min_rate=1, max_rate=100, start_time=0, end_time=783
            )
            generator_stats = [{"rate": w} for w in rates]
            nest.SetStatus(self.pixel_rate_generators, generator_stats)
        else:
            train_spikes, train_spike_times = spike_generator.bellec_spikes(
                train_data, self.n_input_neurons, self.dt
            )
            for ii, ii_spike_gen in enumerate(self.pixel_rate_generators):
                iter_neuron_spike_times = np.multiply(
                    train_spikes[:, ii], train_spike_times
                )
                nest.SetStatus(
                    [ii_spike_gen],
                    {
                        "spike_times": iter_neuron_spike_times[
                            iter_neuron_spike_times != 0
                        ],
                        "spike_weights": [1500.0]
                        * len(iter_neuron_spike_times[iter_neuron_spike_times != 0]),
                    },
                )

    def plot_all(self, gen_idx, idx, save=True, path="."):
        spikes = nest.GetStatus(self.input_spike_detector, keys="events")[0]
        if save:
            visualize.spike_plot(
                spikes, "Input spikes", gen_idx=gen_idx, idx=0, save=save, path=path
            )
            # visualize.plot_data(idx, self.mean_ca_e, self.mean_ca_i,
            #                     self.total_connections_e,
            #                     self.total_connections_i)
            visualize.plot_fr(
                idx=idx, mean_ca_e=self.mean_ca_e, mean_ca_i=self.mean_ca_i, save=save
            )
            # visualize.plot_output(idx, self.mean_ca_e_out)

    def simulate(
        self,
        record_spiking_firingrate,
        train_set,
        targets,
        gen_idx,
        ind_idx,
        sim_idx=0,
        save_plot=False,
        path=".",
        replace_weights=False,
        test=False,
        **kwargs,
    ):
        """
        Simulation method, returns the ex. mean firing rate as model output

        :param record_spiking_firingrate: bool, if the firing rate should be
            recorded
        :param train_set: list, input for the network
        :param targets: list of ints, targets corresponding to `train_set`
        :param gen_idx: int, generation number
        :param ind_idx: int, individual number
        :param sim_idx: int, simulation index
            Default is 0
        :param save_plot: bool, if plots should be saved
            Default is False
        :param path: str, csv file to load the weights and connections
        :param replace_weights, bool, if weights should be loaded, in
            combination with `kwargs` see keyword `weights`
            Default is False
        :param test: bool, Enable if testing the run
            Default is False
        :param kwargs: dict, Dictionary with the weights
               - weights: numpy array of weights, coming from a previous
               simulation to continue the simulation
               requires `replace_weights=True`
        """
        # prepare the connections etc.
        if not test: # True for real simulation runs, i.e. test = False
            self.prepare_network()
            self.connect_internal_bulk()
            self.connect_external_input()
            self.connect_spike_detectors()
            self.connect_noise_bulk()
            self.connect_internal_out()
            self.connect_noise_out()
            self.connect_bulk_to_out()
        #  self.connect_out_to_out()
        # TODO: check replacing operations
        if replace_weights:
            weights = kwargs.get("weights")
            root_dir_path = csv_path
            assert weights.size > 0, "Weights seem to be empty"
            self.replace_weights_(weights, root_dir_path)
        else:
            self.replace_weights(
                path=path,
                typ="eeo",
                source=self.nodes_e,
                target=self.nodes_out_e,
                test=test,
                sim_idx=sim_idx,
                ind_idx=ind_idx
            )
            self.replace_weights(
                path=path,
                typ="eio",
                source=self.nodes_e,
                target=self.nodes_out_i,
                test=test,
                sim_idx=sim_idx,
                ind_idx=ind_idx
            )
            self.replace_weights(
                path=path,
                typ="ieo",
                source=self.nodes_i,
                target=self.nodes_out_e,
                test=test,
                sim_idx=sim_idx,
                ind_idx=ind_idx
            )
            self.replace_weights(
                path=path,
                typ="iio",
                source=self.nodes_i,
                target=self.nodes_out_i,
                test=test,
                sim_idx=sim_idx,
                ind_idx=ind_idx
            )
        # Warm up simulation
        print("Starting simulation")
        self.clear_input()
        nest.Simulate(self.warm_up_time)
        print('Warm up done')
        # start simulation
        model_outs = []
        for idx, target in enumerate(targets):
            # cooling time, empty simulation
            print("Cooling period")
            # Clear input
            self.clear_input()
            nest.Simulate(self.cooling_time)
            print("Cooling done")
            self.clear_records()
            if record_spiking_firingrate:
                self.clear_spiking_events()
            self.set_external_input(
                iteration=gen_idx,
                train_data=train_set[idx],
                target=target,
                path=path,
                save=save_plot,
            )
            sim_steps = np.arange(0, self.t_sim, int(self.record_interval))
            for j, step in enumerate(sim_steps):
                # Do the simulation
                nest.Simulate(self.record_interval)
                if j % 20 == 0:
                    print("Progress: " + str(j / 2) + "%")
                if record_spiking_firingrate:
                    self.record_fr(
                        indx=ind_idx,
                        gen_idx=gen_idx,
                        save=save_plot,
                        record_out=True,
                        path=path,
                    )
                    if len(targets) - 1 == idx: 
                        print("Mean out e ", self.mean_ca_out_e)
                        print("Mean e ", self.mean_ca_e)
                        print('Input spikes ', len(nest.GetStatus(self.input_spike_detector, keys='events')[0]['times']))
                        print('Bulk spikes', len(nest.GetStatus(self.bulks_detector_ex, keys='events')[0]['times']))
                        for n in range(self.n_output_clusters):
                            print('Out spikes', len(nest.GetStatus(self.out_detector_e, keys='events')[n]['times']))
                    self.clear_spiking_events()
                else:
                    self.record_ca(record_out=True)
                # self.record_connectivity()
            print("Simulation loop {} finished successfully".format(idx))
            model_outs.append(self.mean_ca_out_e.copy())
            # clear lists
            self.clear_records()
        # write model_outs
        model_path = os.path.join(path, f"{ind_idx}_{sim_idx}_model_out.npy")
        if os.path.isfile(model_path):
            try:
                os.remove(model_path)
            except FileNotFoundError as fe:
                print(fe, flush=True)
        np.save(model_path, model_outs)
        return model_outs

    @staticmethod
    def replace_weights(source, target, path=".", ind_idx="0", sim_idx="0",
                        typ="e", test=False):
        while True:
            if os.path.isfile(os.path.join(path, f"{ind_idx}_{sim_idx}_weights_{typ}.csv")):
                if os.path.getsize(os.path.join(path,
                                                f"{ind_idx}_{sim_idx}_weights_{typ}.csv")) > 0:
                    break
            else:
                print(f'waiting in replace for weights, {ind_idx}, {sim_idx}')
                time.sleep(2)        
        # Read the connections, i.e. sources and targets
        conns = pd.read_csv(
            os.path.join(path, f"{ind_idx}_{sim_idx}_weights_{typ}.csv"),
            index_col=0)
        # extract the weights and connections
        # sources = conns['source'].values
        # targets = conns['target'].values
        # weights = conns['weights'].values
        print(f"now replacing connection weights `{typ}` of individual {ind_idx} simulation {sim_idx}")
        if not isinstance(source, list):
            source = [source]
        if not isinstance(target, list):
            target = [target]
        for s in source:
            for t in target:
                if test:
                    conn = nest.GetConnections(source=s, target=t)
                    weights = conns[
                        (conns.source.isin(conn.source))
                        & conns.target.isin(conn.target)
                    ].weights.values
                    conn.set(weight=weights)
                else:
                    sources = conns["source"].values
                    targets = conns["target"].values
                    weights = conns["weights"].values
                    syn_spec = {
                        "weight": weights,
                        "synapse_model": "static_synapse",
                        "delay": [1.0] * len(weights),
                    }
                    nest.Connect(
                        pre=sources,
                        post=targets,
                        syn_spec=syn_spec,
                        conn_spec="one_to_one",
                    )

    @staticmethod
    def load_network(path="."):
        conns_eeo = pd.read_csv(os.path.join(path, "{}_connections.csv".format("eeo")))
        conns_eio = pd.read_csv(os.path.join(path, "{}_connections.csv".format("eio")))
        conns_ieo = pd.read_csv(os.path.join(path, "{}_connections.csv".format("ieo")))
        conns_iio = pd.read_csv(os.path.join(path, "{}_connections.csv".format("iio")))
        sources_eeo = conns_eeo["source"].values
        sources_eio = conns_eio["source"].values
        sources_ieo = conns_ieo["source"].values
        sources_iio = conns_iio["source"].values
        targets_eeo = conns_eeo["target"].values
        targets_eio = conns_eio["target"].values
        targets_ieo = conns_ieo["target"].values
        targets_iio = conns_iio["target"].values
        s = np.hstack((sources_eeo, sources_eio, sources_ieo, sources_iio))
        t = np.hstack((targets_eeo, targets_eio, targets_ieo, targets_iio))
        return s, t

    def replace_weights_(self, weights, path="."):
        sources, targets = self.load_network(path)
        print(f"Shape sources {sources.shape}, targets {targets.shape} weights {weights.shape}")
        syn_spec = {
            "weight": weights,
            "synapse_model": "static_synapse",
            "delay": [1.0] * len(weights),
        }
        nest.Connect(
            pre=sources, post=targets, syn_spec=syn_spec, conn_spec="one_to_one"
        )

    @staticmethod
    def save_connections(conn, path=".", typ="e", do_pickle=False):
        # status = nest.GetStatus(conn)
        d = OrderedDict({"source": [], "target": []})
        for c in conn:
            d["source"].extend(c.source)
            d["target"].extend(c.target)
            # d['weight'].append(elem.get('weight'))
        df = pd.DataFrame(d)
        if do_pickle:
            df.to_pickle(os.path.join(path, "{}_connections.pkl".format(typ)))
        df.to_csv(os.path.join(path, "{}_connections.csv".format(typ)))

    def checkpoint(self, ids):
        # Input connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_in))
        f = open("conn_input_{}.bin".format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        # Bulk connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_e))
        f = open("conn_bulke_{}.bin".format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        connections = nest.GetStatus(nest.GetConnections(self.nodes_i))
        f = open("conn_bulki_{}.bin".format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()

        # # Out connections
        connections = nest.GetStatus(nest.GetConnections(self.nodes_out_e[0]))
        f = open("conn_oute_0_{}.bin".format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()
        connections = nest.GetStatus(nest.GetConnections(self.nodes_out_i[0]))
        f = open("conn_outi_0_{}.bin".format(ids), "wb")
        pickle.dump(connections, f, pickle.HIGHEST_PROTOCOL)
        f.close()


if __name__ == "__main__":
    """
    Script to test out different scenarios with the reservoir written in 
    NESTv3.  
    
    For example to run a test execute:
    python reservoir_nest3.py -t -p ../results -i 00 -g -rfr True
    -t: test
    -p: results folder, requires csv files with the network connections
    -i: individual number 
    -g: generation number
    -rfr: record firing rate 
    
    To run an experiment similar to the optimizee execute:
    1. python reservoir_nest3.py -c -p ../results -i 00 -g -rfr True
    2. python reservoir_nest3.py -s -p ../results -i 00 -g -rfr True
    
    """
    def _create_example_dataset(mnist_path="./mnist784_dat/", target_label="0"):
        """
        Creates an example dataset for a given label.
        Returns the data and labels
        """
        from l2l.optimizers.kalmanfilter.data import fetch

        train_set, train_labels, test_set, test_labels = fetch(
            path=mnist_path, labels=target_label
        )
        return train_set, train_labels

    def _save_weights(
        path: str = "./",
        n_simulations: int = 1,
        mu: float = 100.0,
        sigma: float = 50.0,
        ind_idx: int = 0,
        sim_idx: int = 0,
    ):
        """
        Creates random distributed weights for the network and saves them into
        a csv file as `{index}{n_sim}_weights_{typ}.csv`
        """
        for i in range(n_simulations):
            for typ in ["eeo", "eio", "ieo", "iio"]:
                conns = pd.read_csv(
                    os.path.join(csv_path, "{}_connections.csv".format(typ)),
                    index_col=0,
                )
                key = f"{ind_idx}_{sim_idx}_weights_{typ}"
                size = len(conns)
                conns["weights"] = np.random.normal(mu, sigma, size)
                conns.to_csv(os.path.join(path, f"{key}.csv"))


    # init the reservoir (no creation of the network yet)
    reservoir = Reservoir()
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--create", action="store_true",
                        help="Creates the network architecture, should be run once at the beginning",)
    parser.add_argument("-s", "--simulate", action="store_true",
                        help="Simulates the network, should run after the creation",)
    parser.add_argument("-p", "--path", type=str, help="Path to csv files")
    parser.add_argument("-i", "--index", type=str, help="Index of the run")
    parser.add_argument("-si", "--simulation_index", type=str,
                        help="The simulation index to indentify the rank")
    parser.add_argument('-g', '--generation', type=str, help='Generation index')
    parser.add_argument("-rfr", "--record_spiking_firingrate", default=True, type=bool)
    parser.add_argument("-t", "--test", action="store_true",
                        help="To test the reservoir simulation")

    args = parser.parse_args()
    # set the random seed
    np.random.seed(int(args.generation))  
    # comm = MPI.COMM_WORLD
    # index of the actual simulation is obtained from the mpi rank
    # simulation_index = comm.Get_rank()
    simulation_index = int(args.simulation_index)
    if args.create:
        if simulation_index == 0:
            size_eeo, size_eio, size_ieo, size_iio = reservoir.connect_network(args.path)
        # comm.Barrier()

    elif args.simulate:
        csv_path = args.path
        # overall index, something like 12 for generation-individual
        individual_id = int(args.index)
        generation_id = int(args.generation)
        print(f'Generation {generation_id}, Individual {individual_id}', flush=True)
        data = np.load(os.path.join(csv_path, f"{generation_id}_{individual_id}_dataset.npy"),
                       allow_pickle=True).item()
        dataset = data["train_set"]
        labels = data["targets"]
        reservoir.simulate(
            record_spiking_firingrate=args.record_spiking_firingrate,
            path=args.path,
            train_set=dataset,
            targets=labels,
            gen_idx=generation_id,
            ind_idx=individual_id,
            sim_idx=simulation_index,
            test=False,
        )
    elif args.test:
        index = args.index
        csv_path = args.path
        dataset, labels = _create_example_dataset()
        print('creating dataset done')
        labels = np.array(labels).astype(float)
        generation_id = int(args.generation)
        individual_id = int(index[0])
        np.save(os.path.join(csv_path, f'{generation_id}_dataset.npy'), {'train_set': dataset, 'targets': labels})
        # create the connections or load them
        size_eeo, size_eio, size_ieo, size_iio = reservoir.connect_network(csv_path)
        print('network connected')
        # data = pd.read_csv(os.path.join(csv_path, f'{index}_dataset.csv'))
        # dataset = data['train_set'].values
        # labels = data['targets'].values
        # create weights if not existant
        _save_weights(path=csv_path, ind_idx=int(index[0]))
        # load weights
        # weights = np.load('/home/yegenoglu/Documents/toolbox/results_l2l/SNN-EnKF/2021-05-06/weights.npz')['weights']
        # print('weights loaded')
        # create kwargs dictionary
        kwargs = {} # {"weights": weights[-1][int(index[1:])]}
        # print(kwargs.get('weights').shape)
        print("start simulate")
        reservoir.simulate(
            record_spiking_firingrate=args.record_spiking_firingrate,
            # activate test run
            test=True,
            path=args.path,
            train_set=dataset[:2],
            targets=labels[:2],
            # there shouldn't be individuals > 9, so the first index
            # should be only one digit, the rest is simulation index
            # i.e. the ensemble_size
            gen_idx=generation_id,
            ind_idx=individual_id,
            replace_weights=False,
            **kwargs
        )
