"""
Optimizee combining ThaCo and SBI Optimizee
"""

import numpy as np
from os.path import join
import os
import pickle
from .prior import prior, labels, x_obs

from l2l.optimizees.optimizee import Optimizee

class ThaCoOptimizee(Optimizee):
    """
    ...
    """

    def __init__(self, traj):
        """
        ...
        """
        super().__init__(traj)

    def create_individual(self, n=1, prior=prior):
        """
        ...
        """
        samples = prior.sample((n,))
        pop = [dict(zip(labels, sample.tolist())) for sample in samples]
        if n == 1:
            return pop[0], prior # TODO okay?
        return pop, samples

    def bounding_func(self, individual):
        """
        ...
        """
        return individual # prior is already bounded

    def simulate(self, traj):
        """
        ...
        """

        self.ind_idx = traj.individual.ind_idx
        self.generation = traj.individual.generation

        sim_name = f'l2l'
        trial_id = f'gen{self.generation}_ind{self.ind_idx}'
        n_threads = 5 # TODO

        # ref = load_ref('/p/home/jusers/todt1/jusuf/slns/L2L/l2l/optimizees/arbor/ref.csv') # Vergleichskurve ("observation")
        # par = load_params('/p/home/jusers/todt1/jusuf/slns/L2L/l2l/optimizees/arbor/fit.json')
        # opt = ArborParRunner()

        # # replace parameters
        # for label in labels:
        #     par[label] = float(traj.individual[label])

        # ts, um = opt.run(par)

        # # summary statistics
        # obs = self.get_features(ts, um)

        # fitness = -np.mean(np.square(um-ref['U/mV'][:-1])) # mse

        # Paths
        thaco_path = '/home/todt/Dokumente/ThaCo/ThaCo2c-GPU-L2L'
        orig_model_path = join(thaco_path, 'ModelCatalogue/2c3x3')
        this_model_path = join(thaco_path, f'ModelCatalogue/2c3x3_gen{self.generation}_ind{self.ind_idx}') #TODO
        analysis_path = join(thaco_path, 'AnalysisTools/2c')
        this_analysis_config = join(analysis_path, f'analysis_config_gen{self.generation}_ind{self.ind_idx}.yaml')

        # Copy model dir
        os.system(f'cp -r {orig_model_path} {this_model_path}')

        # Prepare tune.yaml
        with open('/home/todt/Dokumente/L2L/l2l/optimizees/thaco/tune.yaml', 'r') as tune_file:
            tune_yaml = tune_file.read()

        out_yaml = tune_yaml.format(WEIGHTINH=traj.individual['inh_weight'],
                                     NOISERATE=traj.individual['noise_rate'],
                                     THWMAX=traj.individual['th_wmax'],
                                     CXWMAX=traj.individual['cx_wmax'],
                                     TIMEPAUSE=round(traj.individual['t_pause_ms'], 1),
                                     TIMEIMG=round(traj.individual['t_img_ms'], 1),
                                     SIMNAME=sim_name,
                                     TRIALID=trial_id)

        with open(join(this_model_path, 'tune.yaml'), 'w') as out_file:
            out_file.write(out_yaml)

        # Run simulation
        # os.system or subprocess.
        os.system(f'cd {thaco_path}; python thaco.py --model_dir {this_model_path} --n_threads {n_threads}')

        # Prepare analysis
        with open('/home/todt/Dokumente/L2L/l2l/optimizees/thaco/analysis_config.yaml', 'r') as analysis_file:
            analysis_yaml = analysis_file.read()

        out_yaml = analysis_yaml.format(SIMNAME=sim_name, TRIALID=trial_id)

        with open(this_analysis_config, 'w') as out_file: # TODO
            out_file.write(out_yaml)

        # Run analysis
        os.system(f'cd {analysis_path}; python 2cAnalysis.py {this_analysis_config}')

        # Read results
        result_path = join(thaco_path, 'Output/Analysis_output/SingleModel', trial_id, 'losses/losses.pkl')
        with open(result_path, 'rb') as result_file:
            results = pickle.load(result_file)

        accuracy = results['losses']['awake_training']['__main__']['class']['Ca-AdEx']['accuracy']
        contrast = results['losses']['awake_training']['__main__']['class']['Ca-AdEx']['L_contrast']

        obs = [accuracy, contrast]
        fitness = -(accuracy-1)**2 # dummy fitness

        return (fitness, obs)
