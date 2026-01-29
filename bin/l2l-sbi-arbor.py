import numpy as np
from l2l.utils.experiment import Experiment

from l2l.optimizees.arbor import ArborOptimizee# OptimizeeParameters
from l2l.optimizers.bayesianinference import SBIOptimizer, SBIOptimizerParameters

from sbi.inference import SNPE

def run_experiment():
    name = 'L2L-SBI'
    experiment = Experiment("/p/scratch/cslns/todt1/L2L/results/")
    # jube_params = { "exec": "python"}

    runner_params = {
        "srun": "srun -n 1 -c 1",
        "exec": "python",
        "max_workers": 40*4
    }

    traj, _ = experiment.prepare_experiment(name=name,
                                            trajectory_name=name,
                                            runner_params=runner_params,
                                            log_stdout=True,
                                            debug=True,
                                            overwrite=False)

    # Optimizee
   #optimizee_parameters = OptimizeeParameters()
    optimizee = ArborOptimizee(traj)#, optimizee_parameters)

    # Optimizer
    optimizer_parameters = SBIOptimizerParameters(pop_size=5000, n_iteration=5, seed=0, save_path='/p/scratch/cslns/todt1/L2L/results/data',
                                                  inference_method=SNPE, restrict_prior=0, x_obs=(-82.6047972031602, 0.2776949999997837, 3.333347222227444, 0.2992816666714235, -69.29435178117059, 10.566454364014849, 0.7262500000117456), tensorboard=False)
    optimizer = SBIOptimizer(traj, optimizee_create_individual=optimizee.create_individual,
                                optimizee_fitness_weights=(1.0, 0.0),
                                parameters=optimizer_parameters,
                                optimizee_bounding_func=optimizee.bounding_func)

    # Run experiment
    experiment.run_experiment(optimizer=optimizer, optimizee=optimizee,
                              optimizer_parameters=optimizer_parameters,
                              )#optimizee_parameters=optimizee_parameters)

    # End experiment
    experiment.end_experiment(optimizer)

def main():
    run_experiment()

if __name__ == '__main__':
    main()
