import logging

import numpy as np
from sdict import sdict

from l2l.optimizees.optimizee import Optimizee
import subprocess
import pickle
import itertools

logger = logging.getLogger("l2l-pse")

# class SubmitJob(gc3libs.Application)

class PSEOptimizee(Optimizee):

    def __init__(self, trajectory, seed=27):

        super(PSEOptimizee, self).__init__(trajectory)
        # If needed
        seed = np.uint32(seed)
        self.random_state = np.random.RandomState(seed=seed)
        self.minnorp0 = -100
        self.maxnorp0 = -1
        self.minnorp1 = -100
        self.maxnorp1 = -1
        # self.minnorp2 = -70
        # self.maxnorp2 = -55
        # self.minnorp3 = 1
        # self.maxnorp3 = 40

    def simulate(self, trajectory):
        self.id = trajectory.individual.ind_idx
        self.p0 = trajectory.individual.p0
        # self.p0 = self.min_max_normalize(np.asarray(self.p0), self.minnorp0, self.maxnorp0, renormalize=True)
        self.p1 = trajectory.individual.p1
        # self.p1 = self.min_max_normalize(np.asarray(self.p1), self.minnorp1, self.maxnorp1, renormalize=True)
        # self.p2 = trajectory.individual.p2
        # self.p2 = self.min_max_normalize(np.asarray(self.p2), self.minnorp2, self.maxnorp2, renormalize=True)
        # self.p3 = trajectory.individual.p3
        # self.p3 = self.min_max_normalize(np.asarray(self.p3), self.minnorp3, self.maxnorp3, renormalize=True)
        # self.p4 = trajectory.individual.p4

        # print('p0', self.p0)
        # print('p1', self.p1)
        # print('p2', self.p2)
        # print('p3', self.p3)
        # print('p4', self.p4)

        import os
        # print("wp", os.getcwd())
        # PROJECT = os.getenv('PROJECT')
        # USER = os.getenv('USER')
        # os.chdir(PROJECT+USER+"/L2L/l2l/optimizees/pse_multi/")
        # print("wp", os.getcwd())

        here = os.path.dirname(os.path.abspath(__file__))
        headerhere = here
        os.chdir(here)

        # Pickle the L2L produced parameters such that your application can pick them up
        # Already make them GPU TVB proof such to pack a single file
        #params = itertools.product(
        #    self.p0,
        #    self.p1,
        #    self.p2,
        #    self.p3,
            # self.p4,
        #)
        # self.p0 = -1*self.p0
        # self.p1 = -1*self.p1
        # params = [self.p0, self.p1, self.p2, self.p3]
        params = [self.p0, self.p1]
        params = np.array([vals for vals in params], np.float32).T
        print('paramsshape', params.shape)
        print('paramsshape', params)
        params_file = open('rateml/sweepars_%d' % self.id, 'wb')
        pickle.dump(params, params_file)
        params_file.close()

        # Start the to optimize process which can be any executable
        # Make sure to read in the pickled data from L2L
        # Set the rateML execution and results folder on your system
        # TODO: make nicer
        try:
            subprocess.run(['python', 'rateml/model_driver_zerlaut.py',
                            # '--model', 'oscillator',
                            '-s0', '32',
                            '-s1', '32',
                            # '-s2', '8',
                            # '-s3', '8',
                            # '-s4', '8',
                            '-n', '5000', '-v', '-sm', '3',
                            # '--tvbn', '76', '--stts', '2',
                            '--procid', str(self.id)], check=True)
        except subprocess.CalledProcessError:
            logger.error('Optimizee process error')

        # Results are dumped to file result_[self.id].txt. Unpickle them here
        self.fitness = []
        cuda_RateML_res_file = open('rateml/result_%d' % self.id, 'rb')
        self.fitness = pickle.load(cuda_RateML_res_file)
        cuda_RateML_res_file.close()
        # self.fitness = np.random.randn(16)
        print("FITNESSSSSSSSSSS", self.fitness.shape)
        # print("FITNESSSSSSSSSSS", len(self.fitness))


        # self.fitness = []
        # if id == 0:
        #     filename = "/p/project/cslns/vandervlag1/L2L/results/Result_0.txt"
        # else:
        #     filename = "/p/project/cslns/vandervlag1/L2L/results/Result_1.txt"
        # with open(filename, "r") as f:
        #     line = f.readline()
        #     while line:
        #         self.fitness.extend([line])
        #         line = f.readline()

        return self.fitness

    def create_individual(self):
        """
        Creates a random value of parameter within given bounds
        """
        # Define the first solution candidate randomly
        # self.bound_gr = [0, 0]  # for delay
        # self.bound_gr[0] = 0
        # self.bound_gr[1] = 94
        #
        # self.bound_gr2 = [0, 0]  # for coupling
        # self.bound_gr2[0] = 0
        # self.bound_gr2[1] = 0.945

        # num_of_parameters = 32  # 48
        # return{'coupling':[5,6,7,8,9], 'delay':[0.1,1,10,12]}
        # delay_array = []
        # coupling_array = []
        # for i in range(num_of_parameters):
        #     delay_array.extend([self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0]])
        #     coupling_array.extend(
        #         [self.random_state.rand() * (self.bound_gr2[1] - self.bound_gr2[0]) + self.bound_gr2[0]])

        # coupling_array = np.linspace(0.003, 0.005, num_of_parameters)
        # delay_array = np.linspace(3.0, 5.0, num_of_parameters)

        # print(delay_array)
        # print(coupling_array)
        # return {'delay': delay_array, 'coupling': coupling_array}
        # return {'delay': self.random_state.rand() * (self.bound_gr[1] - self.bound_gr[0]) + self.bound_gr[0],
        #      'coupling': self.random_state.rand() * (self.bound_gr2[1] - self.bound_gr2[0]) + self.bound_gr2[0]}

        # representing the S, be, ELE, ELI, T parameters Goldman2023. Fitting only be for demo
        # return {'p0': (np.random.rand() * (0 + 100)) + 100}
        # return {'p0': self.min_max_normalize(np.random.uniform(self.minnorp0, self.maxnorp0), self.minnorp0, self.maxnorp0),
        #         'p1': self.min_max_normalize(np.random.uniform(self.minnorp1, self.maxnorp1), self.minnorp1, self.maxnorp1),
                # 'p2': self.min_max_normalize(np.random.uniform(self.minnorp2, self.maxnorp2), self.minnorp2, self.maxnorp2),
                # 'p3': self.min_max_normalize(np.random.uniform(self.minnorp3, self.maxnorp3), self.minnorp3, self.maxnorp3)
                # }

        return {'p0': np.random.uniform(self.minnorp0, self.maxnorp0),
                'p1': np.random.uniform(self.minnorp1, self.maxnorp1),
                }

    @staticmethod
    def min_max_normalize(x, min_val, max_val, renormalize=False):
        if renormalize:
            return x * (max_val - min_val) + min_val
        else:
            return (x - min_val) / (max_val - min_val)

        # representing the (S,) be, ELE, ELI, T parameters Goldman2023. TODO arrayfy
        # return {'p0': (np.random.rand()*(25+100))-25,
        #         'p1': (np.random.rand()*(-55--70))+-55,
        #         'p2': (np.random.rand()*(-55--70))+-55,
        #         'p3': (np.random.rand()*(1+40))+1,
        # #         'p4': (np.random.rand()*(40-5))+5
        #         }
        # for fitting the showcase 1. alpha, beta, gamma, delta scalers for tau_i and Eli. Find out initial values
        # return {'p0': np.random.uniform(0,5,1),
        #         'p1': np.random.uniform(0,5,1),
        #         'p2': np.random.uniform(-60,-80,1),
        #         'p3': np.random.uniform(-60,-80,1),
        #         }

    def bounding_func(self, individual):

        # stay positive
        # if individual['p0'] < 0:
        #     individual['p0'] = -1*individual['p0']
        #
        # # and some stay negative
        # if individual['p1'] > 0:
        #     individual['p1'] = -1*individual['p1']
        #
        # if individual['p2'] > 0:
        #     individual['p2'] = -1*individual['p2']
        #
        # if individual['p3'] > 0:
        #     individual['p3'] = -1 * individual['p3']

        # individual = {
        #     "p0": np.clip(individual['p0'], np.random.uniform(0.00001 ,0.01), np.random.uniform(0.99, 1)),
        #     "p1": np.clip(individual['p1'], np.random.uniform(0.00001 ,0.01), np.random.uniform(0.99, 1)),
            # "p2": np.clip(individual['p2'], np.random.uniform(0.00001 ,0.01), np.random.uniform(0.99, 1)),
            # "p3": np.clip(individual['p3'], np.random.uniform(0.00001 ,0.01), np.random.uniform(0.99, 1))
            #           }

        return individual

        # return {'p0': self.min_max_normalize(np.random.uniform(0, 1), 0, 1),
        #         'p1': self.min_max_normalize(np.random.uniform(-55, -70), -55, -70),
        #         'p2': self.min_max_normalize(np.random.uniform(-55, -70), -55, -70),
        #         'p3': self.min_max_normalize(np.random.uniform(0, 40), 0, 40)
        #         }


def end(self):
    logger.info("End of all experiments. Cleaning up...")
    # There's nothing to clean up though


def main():
    import yaml
    import os
    import logging.config

    from ltl import DummyTrajectory
    from ltl.paths import Paths
    from ltl import timed

    # TODO: Set root_dir_path here
    paths = Paths('pse', dict(run_num='test'), root_dir_path='.')  # root_dir_path='.'

    fake_traj = DummyTrajectory()
    optimizee = PSEOptimizee(fake_traj)
    # ind = Individual(generation=0,ind_idx=0,params={})
    params = optimizee.create_individual()
    # params['generation']=0
    params['ind_idx'] = 0
    # fake_traj.f_expand(params)
    # for key,val in params.items():
    #    ind.f_add_parameter(key, val)
    fake_traj.individual = sdict(params)
    # fake_traj.individual.ind_idx = 0

    testing_error = optimizee.simulate(fake_traj)
    print("Testing error is ", testing_error)


if __name__ == "__main__":
    main()
