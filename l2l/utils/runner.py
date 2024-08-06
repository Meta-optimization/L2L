import os.path
import pickle
import logging
import shlex, subprocess
import time
import copy
import zipfile
from l2l.utils.trajectory import Trajectory

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
        exit_codes = self.simulate_generation(generation, n_inds)

        ## Touch done generation
        logger.info("Finished generation: " + str(self.generation))

        if all(exit_code == 0 for exit_code in exit_codes):
            results = self.collect_results_from_run(generation, self.trajectory.individuals[generation])
        else:
            # not all individuals finished without error (even potentially after restarting)
            raise RuntimeError(f"Generation {generation} did not finish successfully")
        #create zipfiles for Err and Out files 
        if self.srun_command:
            self.create_zipfile(self.work_paths["individual_logs"], f"logs_generation_{generation}")


        return results

    def produce_run_command(self, gen, idx):
        if self.srun_command:
            # HPC case with slurm
            run_ind = f"{self.srun_command} --output={self.work_paths['individual_logs']}/out_{gen}_{idx}.log --error={self.work_paths['individual_logs']}/err_{gen}_{idx}.log {self.exec_command} {gen} {idx} &"
        else:
            # local case without slurm
            run_ind = f"{self.exec_command} {gen} {idx} > {self.work_paths['individual_logs']}/out_{gen}_{idx}.log 2> {self.work_paths['individual_logs']}/err_{gen}_{idx}.log &"
            # TODO output redirection via > and 2> doesnt work
        logger.info(f"{run_ind}")
        args = shlex.split(f"{run_ind}")

        logger.info(args)
        return args
    

    def simulate_generation(self, gen, n_inds):
        """
        Executes n_inds srun commands, waits for them to finish and writes their exit codes to 'exit_codes.log'
        """
        
        running_individuals = {}
        finished_individuals = {}
        for idx in range(n_inds):
            args = self.produce_run_command(gen, idx)
            process = subprocess.Popen(args)
            running_individuals[idx] = process

        # Wait for all individual to finish
        # Restart failed individuals 
        retry=0
        while True:

            for idx in list(running_individuals.keys()):
                process = running_individuals[idx]
                status_code = process.poll()

                #print(f"status {idx}: {status_code}")
                
                if status_code == None:
                    # indivdual still running
                    continue
                elif status_code == 0:
                    print(f"status {idx}: {status_code}")
                    # individual finished without error
                    finished_individuals[idx] = running_individuals.pop(idx)
                else: 
                    print(f"status {idx}: {status_code} {status_code==None}")
                    # individual raised error
                    # TODO depending on what kind of error restart failed individual
                    # TODO pass reference to optimizer from environment.py and call optimizer.restart(ind)
                    if status_code > 128 and retry<20:#Error spawning step, wait a bit?
                        print(f"Restarting {idx} from error {status_code}\n retry {retry}")
                        time.sleep(4)
                        args = self.produce_run_command(gen, idx)
                        process = subprocess.Popen(args)
                        running_individuals[idx] = process
                        retry += 1
                    else:
                        raise NotImplementedError("restart failed individual")

            if not running_individuals:
                # all processes finished
                break
            time.sleep(5)
        
        sorted_exit_codes = [finished_individuals[idx].poll() for idx in range(n_inds)]
        return sorted_exit_codes





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
                'import gc\n' +
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
                'handle_res.close()\n' + 
                'gc.collect()')
        f.close()

    def dump_traj(self, trajectory):
        """Dumpes trajectory files.
        :param trajectory, object to be dumped"""
        #trajfname = "trajectory_%s.bin" % (trajectory.individual.generation)
        #handle = open(os.path.join(self.work_paths["trajectories"], trajfname),
        #                    "wb")
        #pickle.dump(trajectory, handle, pickle.HIGHEST_PROTOCOL)
        #handle.close()

        mtrajfname = os.path.join(self.work_paths["trajectories"], "whole_trajectory.bin")
        mtraj = trajectory
        if os.path.isfile(mtrajfname):
            with open(mtrajfname, 'rb') as mhandle:
                mtraj = pickle.load(mhandle)
                mtraj.individuals[trajectory.individual.generation] = trajectory.individuals[trajectory.individual.generation]

        with open(mtrajfname, 'wb') as mhandle:
            pickle.dump(mtraj, mhandle, pickle.HIGHEST_PROTOCOL)

        tmpgen = trajectory.individuals[trajectory.individual.generation]
        tmptraj = Trajectory()
        tmptraj.individual = trajectory.individual
        tmptraj.individuals[trajectory.individual.generation] = tmpgen
        trajfname = "trajectory_%s.bin" % (trajectory.individual.generation)
        handle = open(os.path.join(self.work_paths["trajectories"], trajfname),
                            "wb")
        pickle.dump(tmptraj, handle, pickle.HIGHEST_PROTOCOL)
        handle.close()
        del tmptraj

    def create_zipfile(self, folder, filename):
        """
        Creates zipfile and deletes files included in the zip file
        :param folder: path to folder containing the files
        :param filename: filename of the created zip file
        """
        # Full path for the zip file
        zip_path = os.path.join(folder, filename + '.zip')

        # Creating the zip file in the specified folder
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as target:
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if file.endswith('.log'):
                        add = os.path.join(root, file)
                        target.write(add, os.path.relpath(add, folder))
                        # Deleting the log files after zipping
                        os.remove(add)

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
