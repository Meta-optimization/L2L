import numpy as np
from l2l.utils.experiment import Experiment

from l2l.optimizees.bayesianinference import TestSBIOptimizee, TestSBIOptimizeeParameters
from l2l.optimizers.bayesianinference import SBIOptimizer, SBIOptimizerParameters

from sbi.inference import SNPE

def run_experiment():
    name = 'L2L-SBI'
    experiment = Experiment("../results/")
    jube_params = { "exec": "python"}

    traj, all_jube_params = experiment.prepare_experiment(name=name,
                                                          trajectory_name=name,
                                                          jube_parameter=jube_params,
                                                          log_stdout=True)

    # Optimizee
    optimizee_parameters = TestSBIOptimizeeParameters(sim_type='invalid')
    optimizee = TestSBIOptimizee(traj, optimizee_parameters)

    # Optimizer
    optimizer_parameters = SBIOptimizerParameters(pop_size=50, n_iteration=2, seed=0, save_path='/home/todt/Dokumente/L2L/results/data',
                                                  inference_method=SNPE, restrict_prior=0, x_obs=[0.5, 0.5])
    optimizer = SBIOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                optimizee_fitness_weights=(0.5, 0.5),
                                parameters=optimizer_parameters,
                                optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              optimizee_parameters=optimizee_parameters)

    # End experiment
    experiment.end_experiment(optimizer)

def main():
    run_experiment()

if __name__ == '__main__':
    main()
