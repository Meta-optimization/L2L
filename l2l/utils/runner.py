import os.path
import pickle
import logging
import subprocess

logger = logging.getLogger("utils.runner")

class Runner():
    """
    """

    def __init__(self, trajectory, gen):
        """
        :param trajectory: A trajectory object holding the parameters to use in the initialization
        :param gen: number of the generation
        """
        self.trajectory = trajectory
        self.generation = gen

        args = self.trajectory.parameters["runner_params"].params
        self.path = args['paths_obj'].simulation_path
        self.srun_command = args['srun']
        self.exec_command = args['exec']



        # Create directories for workspace
        subdirs = ['trajectories', 'results', 'individual_logs']
        self.work_paths = {sdir: os.path.join(self.path, sdir) for sdir in subdirs}
        os.makedirs(self.path, exist_ok=True)
        for dir in self.work_paths:
            os.makedirs(self.work_paths[dir], exist_ok=True)

        self.optimizeepath = os.path.join(self.path, "optimizee.bin")
        self.debug_stderr = self.trajectory.debug
        self.stop_run = self.trajectory.stop_run
        self.timeout = self.trajectory.timeout

    def collect_results_from_run(self, generation, individuals):
        """
        Collects the results generated by each individual in the generation. Results are, for the moment, stored
        in individual binary files.
        :param generation: generation id
        :param individuals: list of individuals which were executed in this generation
        :return results: a list containing objects produced as results of the execution of each individual
        """
        results = []
        for ind in individuals:
            indfname = "results_%s_%s.bin" % (generation, ind.ind_idx)
            handle = open(os.path.join(self.work_paths["results"], indfname), "rb")
            results.append((ind.ind_idx, pickle.load(handle)))
            handle.close()

        return results

    def run(self, trajectory, generation):
        """
        Takes care of running the generation by executing run_optimizee.py in parallel, waiting for the execution and gathering the results.
        :param trajectory: trajectory object storing individual parameters for each generation
        :param generation: id of the generation
        :return results: a list containing objects produced as results of the execution of each individual
        """
        self.prepare_run_file()

        # Dump trajectory for each optimizee run in the generation
        # each trajectory needs an individual to get the correct generation
        trajectory.individual = self.trajectory.individuals[generation][0]
        self.dump_traj(trajectory)

        logger.info("Running generation: " + str(self.generation))

        n_inds = len(trajectory.individuals[generation])
        self.simulate_generation(generation, n_inds)

        ## Touch done generation
        logger.info("Finished generation: " + str(self.generation))


        #TODO read exit codes before trying to collect results

        results = self.collect_results_from_run(generation, self.trajectory.individuals[generation])
        return results
    

    def simulate_generation(self, gen, n_inds):
        """
        Executes n_inds srun commands, waits for them to finish and writes their exit codes to 'exit_codes.log'
        """

        if self.srun_command:
            # HPC case with slurm
            run_ind = f"{self.srun_command} --output={self.work_paths['individual_logs']}/out_{gen}_$idx.log --error={self.work_paths['individual_logs']}/out_{gen}_$idx.log {self.exec_command}"
        else:
            # local case without slurm
            run_ind = self.exec_command

        script_content = f"""#!/bin/bash
pids=()
exit_codes=()

echo "gen: {gen}"
echo "n_inds: {n_inds}"

for idx in $(seq 0 $(({n_inds}-1)))
do
    {run_ind} {gen} $idx &
    pid=$!
    pids+=($pid)
    echo "Started srun for idx=$idx with PID $pid"
done

echo "PIDs: ${{pids[@]}}"

# Wait for all background jobs to complete and capture their exit codes
for pid in "${{pids[@]}}"
do
    echo "Waiting for PID $pid"
    wait $pid
    exit_code=$?
    exit_codes+=($exit_code)
    echo "PID $pid exited with code $exit_code"
done

# Write exit codes to log file
echo "Exit codes: ${{exit_codes[@]}}"
echo "Exit codes: ${{exit_codes[@]}}" > {self.work_paths['individual_logs']}/exit_codes.log

    """

        script_command = f"bash <<'EOF'\n{script_content}\nEOF"
        result = subprocess.run(script_command, shell=True, capture_output=True, text=True)
        
        # TODO use logger.info()
        print("srun command out:", result.stdout)
        print("srun command err:", result.stderr)





    def prepare_run_file(self):
        """
        Writes a python run file which takes care of loading the optimizee from a binary file, the trajectory object
        of each individual. Then executes the 'simulate' function of the optimizee using the trajectory and
        writes the results in a binary file.
        :param path_ready: path to store the ready files
        :return true if all files are present, false otherwise
        """
        trajpath = os.path.join(self.work_paths["trajectories"],
                                'trajectory_" + str(iteration) + ".bin')
        respath = os.path.join(self.work_paths['results'],
                               'results_" + str(iteration) + "_" + str(idx) + ".bin')
        f = open(os.path.join(self.path, "run_optimizee.py"), "w")
        f.write('import pickle\n' +
                'import sys\n' +
                'iteration = sys.argv[1]\n' +
                'idx = sys.argv[2]\n' +
                'handle_trajectory = open("' + trajpath + '", "rb")\n' +
                'trajectory = pickle.load(handle_trajectory)\n' +
                'handle_trajectory.close()\n' +
                'handle_optimizee = open("' + self.optimizeepath + '", "rb")\n' +
                'optimizee = pickle.load(handle_optimizee)\n' +
                'handle_optimizee.close()\n\n' +
                'trajectory.individual = trajectory.individuals[int(iteration)][int(idx)] \n'+
                'res = optimizee.simulate(trajectory)\n\n' +
                'handle_res = open("' + respath + '", "wb")\n' +
                
                'pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)\n' +
                'handle_res.close()\n\n')
                # TODO remove bottom two lines?
        f.close()

    def dump_traj(self, trajectory):
        """Dumpes trajectory files.
        :param trajectory, object to be dumped"""
        trajfname = "trajectory_%s.bin" % (trajectory.individual.generation)
        handle = open(os.path.join(self.work_paths["trajectories"], trajfname),
                            "wb")
        pickle.dump(trajectory, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()


def prepare_optimizee(optimizee, path):
    """
    Helper function used to dump the optimizee it a binary file for later loading during run.
    :param optimizee: the optimizee to dump into a binary file
    :param path: The path to store the optimizee.
    """
    # Serialize optimizee object so each process can run simulate on it independently on the CNs
    fname = os.path.join(path, "optimizee.bin")
    f = open(fname, "wb")
    pickle.dump(optimizee, f)
    f.close()
    logger.info("Serialized optimizee writen to path: {}".format(fname))
