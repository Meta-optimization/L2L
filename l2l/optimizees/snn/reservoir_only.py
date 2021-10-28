import json
import nest
import numpy as np

from l2l.optimizees.snn import spike_generator, visualize


class ReservoirNetwork:
    def __init__(self):
        with open('config.json', 'rb') as jsonfile:
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

    def create_network(self):
        """Helper functions to create the network"""
        self.reset_kernel()
        self.create_synapses()
        self.create_nodes()
        self.create_input_spike_detectors()
        self.pixel_rate_generators = self.create_pixel_rate_generator(
            self.input_type)
        self.noise = nest.Create("poisson_generator")
        nest.PrintNodes()

    def connect_network(self):
        """
        Connection routines
        - Input to bulk
        - Bulk to bulk
        - Bulk to out
        - Out to out
        - Spike detectors to input, bulk, out
        - Noise to out
        - Noise to bulk
        """
        # Do the connections
        self.connect_internal_bulk()
        self.connect_external_input()
        self.connect_spike_detectors()
        self.connect_noise_bulk()
        self.connect_internal_out()
        self.connect_bulk_to_out()
        self.connect_noise_out()
        # self.connect_out_to_out()

    def reset_kernel(self):
        nest.ResetKernel()
        nest.set_verbosity("M_ERROR")
        nest.local_num_threads = int(self.config['threads'])
        nest.rng_seed = int(self.config["seed"])
        nest.resolution = self.dt
        nest.overwrite_files = True

    @staticmethod
    def create_synapses():
        nest.CopyModel("static_synapse", "random_synapse")
        nest.CopyModel("static_synapse", "random_synapse_i")

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
            # FIXME changed to len(rates) from len(offsets)
            self.pixel_rate_generators = nest.Create("poisson_generator", len(rates))

    def connect_internal_bulk(self):
        syn_dict_e = {
            "synapse_model": "random_synapse",
            "weight": nest.random.normal(self.psc_e, 50.0),
        }
        syn_dict_i = {
            "synapse_model": "random_synapse_i",
            "weight": nest.random.normal(self.psc_i, 50.0),
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
        poisson_gen = nest.Create(
            "poisson_generator",
            1,
            {"rate": 10000.0},
        )
        syn_dict = {"synapse_model": "static_synapse", "weight": 1}
        syn_dict_i = {"synapse_model": "static_synapse", "weight": 1}
        nest.Connect(poisson_gen, self.nodes_e, "all_to_all", syn_spec=syn_dict)
        nest.Connect(poisson_gen, self.nodes_i, "all_to_all", syn_spec=syn_dict_i)

    def connect_internal_out(self):
        # Connect out
        conn_dict = {"rule": "fixed_indegree", "indegree": 2, "allow_multapses": False}
        syn_dict = {"synapse_model": "random_synapse"}
        conn_dict_i = {
            "rule": "fixed_indegree",
            "indegree": 2,
            "allow_multapses": False,
        }
        syn_dict_i = {"synapse_model": "random_synapse_i"}
        for ii in range(self.n_output_clusters):
            nest.Connect(
                self.nodes_out_e[ii], self.nodes_out_e[ii], conn_dict, syn_spec=syn_dict
            )
            nest.Connect(
                self.nodes_out_e[ii], self.nodes_out_i[ii], conn_dict, syn_spec=syn_dict
            )
            nest.Connect(
                self.nodes_out_i[ii],
                self.nodes_out_e[ii],
                conn_dict_i,
                syn_spec=syn_dict_i,
            )
            nest.Connect(
                self.nodes_out_i[ii],
                self.nodes_out_i[ii],
                conn_dict_i,
                syn_spec=syn_dict_i,
            )

    def connect_bulk_to_out(self):
        # Bulk to out
        conn_dict_e = {
            "rule": "fixed_indegree",
            # 0.3 * self.number_out_exc_neurons
            "indegree":  int(self.n_bulk_ex_neurons/(self.n_neurons_out_e/4)),
            "allow_multapses": False,
            "allow_autapses": False,
        }
        conn_dict_i = {
            "rule": "fixed_indegree",
            # 0.2 * self.number_out_exc_neurons
            "indegree":  int(self.n_bulk_in_neurons/(self.n_neurons_out_e/4)),
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
            nest.Connect(
                self.nodes_e, self.nodes_out_e[j], conn_dict_e, syn_spec=syn_dict_e
            )
            nest.Connect(
                self.nodes_e, self.nodes_out_i[j], conn_dict_e, syn_spec=syn_dict_e
            )
            nest.Connect(
                self.nodes_i, self.nodes_out_e[j], conn_dict_i, syn_spec=syn_dict_i
            )
            nest.Connect(
                self.nodes_i, self.nodes_out_i[j], conn_dict_i, syn_spec=syn_dict_i
            )

    def connect_noise_out(self):
        poisson_gen = nest.Create(
            "poisson_generator",
            1,
            {"rate": 10000.0},
        )
        syn_dict = {"synapse_model": "static_synapse", "weight": 1}
        syn_dict_i = {"synapse_model": "static_synapse", "weight": 1}
        for j in range(self.n_output_clusters):
            nest.Connect(
                poisson_gen, self.nodes_out_e[j], "all_to_all", syn_spec=syn_dict
            )
            nest.Connect(
                poisson_gen, self.nodes_out_i[j], "all_to_all", syn_spec=syn_dict_i
            )

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

    def connect_greyvalue_input(self):
        """Connects input to bulk"""
        indegree = 8
        weight = 100.
        syn_dict_e = {
            "synapse_model": "random_synapse",
            #                                    size=weights_len_e)}
            "weight": nest.random.normal(self.psc_e, weight),
        }
        syn_dict_i = {
            "synapse_model": "random_synapse_i",
            # "weight": nest.random.normal(1, 2),
        }
        syn_dict_input = {
            "synapse_model": "random_synapse",
            "weight": nest.random.normal(self.psc_e, weight),
        }
        nest.Connect(
            self.pixel_rate_generators,
            self.nodes_in,
            "one_to_one",
            syn_spec=syn_dict_input,
        )
        # connect input to bulk
        conn_dict = {
            "rule": "fixed_indegree",
            "indegree": indegree,
            "allow_autapses": False,
            "allow_multapses": False,
        }
        nest.Connect(
            self.nodes_in,
            self.nodes_e,
            conn_spec=conn_dict,  # all_to_all
            syn_spec=syn_dict_e,
        )
        nest.Connect(
            self.nodes_in,
            self.nodes_i,
            conn_spec=conn_dict,  # all_to_all
            syn_spec=syn_dict_i,
        )

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
            spike_in = nest.GetStatus(self.input_spike_detector, keys='events')[0]
            visualize.spike_plot(
                spike_in,
                "Input spikes",
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

    def clear_input(self):
        """
        Sets a very low rate to the input, for the case where no input is
        provided
        """
        generator_stats = [{"rate": 1.0} for _ in range(self.n_input_neurons)]
        nest.SetStatus(self.pixel_rate_generators, generator_stats)

    def clear_records(self):
        """
        Empties lists or recreates them, clears the input spike detector
        """
        self.mean_ca_i.clear()
        self.mean_ca_e.clear()
        self.mean_ca_out_e.clear()
        self.mean_ca_out_e = [[] for _ in range(self.n_output_clusters)]
        self.mean_ca_out_i.clear()
        self.mean_ca_out_i = [[] for _ in range(self.n_output_clusters)]
        nest.SetStatus(self.input_spike_detector, {"n_events": 0})

    def clear_spiking_events(self):
        """
        Clears spiking events in the detectors
        """
        nest.SetStatus(self.bulks_detector_ex, "n_events", 0)
        nest.SetStatus(self.bulks_detector_in, "n_events", 0)
        for i in range(self.n_output_clusters):
            nest.SetStatus(self.out_detector_e[i], "n_events", 0)
            nest.SetStatus(self.out_detector_i[i], "n_events", 0)

    def simulate(self, record_spiking_firingrate, train_set, targets, gen_idx,
                 ind_idx, path='.', save_plot=False, with_data=True):
        """
        Simulation method, returns the ex. mean firing rate as model output

        :param record_spiking_firingrate: bool, if the firing rate should be
            recorded
        :param train_set: list, input for the network
        :param targets: list of ints, targets corresponding to `train_set`
        :param gen_idx: int, generation number
        :param ind_idx: int, individual number
        :param path: str, path route to save the plots
        :param save_plot: bool, if plots should be saved
        :param with_data: bool, if the external input should be set with the data
        """
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
            if with_data:
                self.set_external_input(
                    iteration=gen_idx,
                    train_data=train_set[idx],
                    target=target,
                    path=path,
                    save=save_plot,
                )
            # run the simulation
            sim_steps = np.arange(0, self.t_sim, self.record_interval)
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
                else:
                    self.record_ca(record_out=True)
            print("Simulation loop {} finished successfully".format(idx))
            # print(nest.GetStatus(self.out_detector_e, keys='events'))
            print("Mean out e ", self.mean_ca_out_e)
            # print("Mean e ", self.mean_ca_e)
            print(f"Mean out i {self.mean_ca_out_i}")
            model_outs.append(self.mean_ca_out_e.copy())
            print('Input spikes ', len(nest.GetStatus(self.input_spike_detector, keys='events')[0]['times']))
            print('Bulk spikes', len(nest.GetStatus(self.bulks_detector_ex, keys='events')[0]['times']))
            print('Out spikes', len(nest.GetStatus(self.out_detector_e, keys='events')[0]['times']))
            print('Out spikes', len(nest.GetStatus(self.out_detector_e, keys='events')[1]['times']))
            # clear lists
            self.clear_records()
        return model_outs


if __name__ == '__main__':

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

    # load config
    with open('config.json', 'rb') as jsonfile:
        config = json.load(jsonfile)
    rng = np.random.default_rng(config['seed'])
    # create the dataset
    dataset, labels = _create_example_dataset()
    # Reservoir init
    reservoir = ReservoirNetwork()
    reservoir.create_network()
    reservoir.connect_network()
    model_outs = reservoir.simulate(record_spiking_firingrate=True,
                                    train_set=dataset[:1],
                                    targets=labels[:1],
                                    gen_idx=0,
                                    ind_idx=0,
                                    path='.',
                                    save_plot=False,
                                    with_data=True)
