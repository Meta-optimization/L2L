from l2l.utils.trajectory import Trajectory
import logging
from l2l.utils.runner import Runner

logger = logging.getLogger("utils.environment")


class Environment:
    """
    The Environment class takes the place of the pypet Environment and provides the required functionality
    to execute the inner loop.
    Based on the pypet environment concept: https://github.com/SmokinCaterpillar/pypet
    """

    def __init__(self, *args, **keyword_args):
        """
        Initializes an Environment
        :param args: arguments passed to the environment initialization
        :param keyword_args: arguments by keyword. Relevant keywords are trajectory and filename.
        The trajectory object holds individual parameters and history per generation of the exploration process.
        """
        if 'trajectory' in keyword_args:
            self.trajectory = Trajectory(name=keyword_args['trajectory'], debug = keyword_args['debug'],
                                         stop_run = keyword_args['stop_run'], timeout=keyword_args['timeout'])
        if 'checkpoint' in keyword_args:
            self.trajectory = keyword_args["checkpoint"]
            self.trajectory.is_loaded = True
        if 'filename' in keyword_args:
            self.filename = keyword_args['filename']
        self.postprocessing = None
        self.multiprocessing = True
        if 'multiprocessing' in keyword_args:
            self.multiprocessing = keyword_args['multiprocessing']
        self.run_id = 0
        self.enable_logging()

    def run(self):
        """
        Runs all generations of the optimizees using the runner.
        """
        result = {}
        logger.info(f"Environment run starting Runner for n iterations: {self.trajectory.par['n_iteration']}")
        runner = Runner(self.trajectory, self.trajectory.par['n_iteration']+self.trajectory.individual.generation)
        for it in range(self.trajectory.individual.generation, self.trajectory.par['n_iteration']+self.trajectory.individual.generation):
            if self.multiprocessing:
                # Multiprocessing is done through the runner
                result[it] = []
                logger.info(f"Iteration: {it+1}/{self.trajectory.par['n_iteration']}")
                # execute run
                try:
                    result[it] = runner.run(self.trajectory,it)
                except Exception as e:
                    if self.logging:
                        logger.exception("Error launching run: " + str(e.__cause__))
                    raise e

            # Add results to the trajectory
            self.trajectory.results.f_add_result_to_group("all_results", it, result[it])
            self.trajectory.current_results = result[it]
            # Update trajectory file
            runner.dump_traj(self.trajectory)
            # Perform the postprocessing step in order to generate the new parameter set
            self.postprocessing(self.trajectory, result[it])
        runner.close_workers()

    def add_postprocessing(self, func):
        """
        Function to add a postprocessing step
        :param func: the function which performs the postprocessing. Postprocessing is the step where the results
        are assessed in order to produce a new set of parameters for the next generation.
        """
        self.postprocessing = func

    def enable_logging(self):
        """
        Function to enable logging
        TODO think about removing this.
        """
        self.logging = True

    def disable_logging(self):
        """
        Function to enable logging
        """
        self.logging = False
