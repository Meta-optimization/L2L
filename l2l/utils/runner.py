import os
import os.path
import pickle
import logging
import shlex, subprocess
import time
import copy
import zipfile
import sys
  
from l2l.utils.trajectory import Trajectory

logger = logging.getLogger("utils.runner")

class Runner():
    """
    A class used to launch the individual optimizees in parallel within the available computing resources. Takes care of launching, monitoring and ending the execution of the individuals. It generates workers which are assigned one or more individuals to be executed within each generation. It also takes care of collecting the results and relaunching individuals if there are runtime or logic errors associated with the execution.

    ...

    Methods
    -------
    collect_results_from_run(self, generation, individuals)
    run(self, trajectory, generation)
    produce_run_command(idx)
    launch(idx)
    launch_workers()
    close_workers()
    restart_worker(w_id)
    simulate_generation(gen, n_inds)
    prepare_run_file()
    dump_traj(trajectory)
    create_zipfile(folder, filename)
    """

    def __init__(self, trajectory, iterations):
        """
        :param trajectory: A trajectory object holding the parameters to use in the initialization
        :param iterations: number of iterations in the optimization process
        """
        self.trajectory = trajectory
        self.iterations = iterations

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


        self.pending_individuals = []
        self.running_individuals = []
        self.finished_individuals = []

        self.running_workers = {}
        self.running_workers_individual_indeces = {}  # TODO built a cleaner data structure for the workers that stores both its process and the idx of the current individual
        self.idle_workers = {}

        self.outputpipes = {}
        self.inputpipes = {}

        self.n_inds = len(trajectory.individuals[0])
        self.n_workers = min(self.n_inds, args['max_workers'])
        

        self.prepare_run_file()
        self.launch_workers()
        logger.info(f"{self.n_inds} workers launched\n")


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
        self.generation = generation
        #self.prepare_run_file()
        # Dump trajectory for each optimizee run in the generation
        # each trajectory needs an individual to get the correct generation
        trajectory.individual = self.trajectory.individuals[generation][0]
        self.dump_traj(trajectory)

        logger.info("Running generation: " + str(self.generation))

        exit_codes = self.simulate_generation(self.generation, len(trajectory.individuals[generation]))

        ## Touch done generation
        logger.info("Finished generation: " + str(self.generation))

        if all(exit_code == 0 for exit_code in exit_codes):
            results = self.collect_results_from_run(generation, self.trajectory.individuals[generation])
        else:
            # not all individuals finished without error (even potentially after restarting)
            raise RuntimeError(f"Generation {generation} did not finish successfully")
        #create zipfiles for Err and Out files 
        self.create_zipfile(self.work_paths["individual_logs"], f"logs_generation_{generation}")

        return results

    def produce_run_command(self, idx):
        """
        Generates a string that can be used to launch an instance of the optimizee with specific parameters, also called an individual.
        :param idx: the id of the individual to be launched.
        """
        log_files = {'stdout': os.path.join(self.work_paths['individual_logs'], f'out_{idx}.log'),
                     'stderr': os.path.join(self.work_paths['individual_logs'], f'err_{idx}.log')} 
        if self.srun_command:
            # HPC case with slurm
            log_files ={}
            run_ind = f"{self.srun_command} --output={self.work_paths['individual_logs']}/out_{idx}.log --error={self.work_paths['individual_logs']}/err_{idx}.log {self.exec_command} {idx} &"
        else:
            # local case without slurm
            run_ind = f"{self.exec_command} {idx}"
        args = shlex.split(run_ind)

        logger.info(f"{run_ind}")
        return args, log_files

    def launch_worker(self, w_id):
        """
        This function uses the subprocess.Popen function to launch the command required to initialize a parallel worker. This worker will stay alive during the whole optimization run and will receive the id of the individuals it will execute each generation. It also generates the pipes (files in a shared file system) to communicate between the runner and the worker.
        Each worker has its own file in the path 'individual_logs' with the worker_ prefix.
        The logs pertaining to each individual are also stored in 'individual_logs' with err_ and out_ prefixes.
        :param idx: the id of the individual to be launched.
        """
        outputpipename = os.path.join(self.work_paths['individual_logs'],f"outputpipe_{w_id}")
        open(outputpipename, 'w+').close()
        logger.info(f"Pipe created {outputpipename}")
        inputpipename = os.path.join(self.work_paths['individual_logs'],f"inputpipe_{w_id}")
        open(inputpipename, 'w+').close()
        logger.info(f"Pipe created {inputpipename}")
        args, log_files = self.produce_run_command(w_id)
        if log_files:
            process = subprocess.Popen(args, stdout=open(log_files['stdout'], 'w'), stderr=open(log_files['stderr'], 'w'))
        else: 
            process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)#, stdin=subprocess.PIPE)
        try:
            self.outputpipes[w_id] = open(outputpipename, 'r')
            self.inputpipes[w_id] = open(inputpipename, 'w')
        except Exception as e:
            logger.info(f"{e}")
        #os.set_blocking(self.outputpipes[idx].fileno(), False)
        self.idle_workers[w_id] = process
        logger.info(f"Worker created {w_id}")
    
    def launch_workers(self):
        """
        Takes care of launching enough workers as allowed by the available computing resources.
        """
        for w_id in range(self.n_workers):
            self.launch_worker(w_id)
        logger.info(f"All {self.n_workers} workers created")


    def close_workers(self):
        """
        Makes sure all workers are notified that the optimization run is over and closes all open pipes and files.
        """
        for w_id in range(self.n_workers):
            self.inputpipes[w_id].write(f"0 0 0\n")#.encode('ascii'))
            self.inputpipes[w_id].flush()
            self.outputpipes[w_id].close()
    
    def restart_worker(self, w_id):
        """
        Takes care of handling the restart of a worker and its associated individual which failed by any reason, either runtime or logic.
        :param w_id: the id of the worker to be launched.
        """
        self.running_workers.pop(w_id)
        self.launch_worker(w_id)

    def restart_individual(self, gen, idx):
        """
        Takes care of handling the restart of an individual which failed by any reason, either runtime or logic.
        :param gen: the current generation
        :param idx: the id of the individual to be launched.i
        """
        #TODO: implement different restart strategies depending on the optimizee.
        self.pending_individuals.append(idx)


    def populate_free_workers(self, gen):
        # assing pending inds to free workers
        while self.idle_workers and self.pending_individuals:
            w_id = list(self.idle_workers.keys())[0]
            next_idx = self.pending_individuals.pop(0)

            self.inputpipes[w_id].write(f"{gen} {next_idx} 1\n")#.encode('ascii'))
            self.inputpipes[w_id].flush()

            self.running_workers[w_id] = self.idle_workers.pop(w_id)
            self.running_workers_individual_indeces[w_id] = next_idx
            self.running_individuals.append(next_idx)
            print(f"--- sent idx {next_idx} to worker {w_id}")



    def simulate_generation(self, gen, n_inds):
        """
        Executes n_inds srun commands, waits for them to finish and writes their exit codes to 'exit_codes.log'
        :param gen: the current generation
        :param n_inds: the number of individuals within the generation
        """

        self.pending_individuals = list(range(n_inds))
        self.populate_free_workers(gen=gen)


        # Wait for all individual to finish
        # Restart failed individuals 
        retry=0
        sorted_exit_codes = [1]*n_inds
        logger.info(f"All workers started running individuals for gen {gen}\n")
        # Add a try catch block to manage restarting individuals correctly
        logger.info(f"Reading output from gen {gen}")
        while True:

            for w_id in list(self.running_workers.keys()):
                process = self.running_workers[w_id]
                ind_idx = self.running_workers_individual_indeces[w_id]
                status_code = process.poll()
                try:
                    out = self.outputpipes[w_id].readline().replace('\n', '')
                except Exception as e: 
                    logger.error(f"Exception: {e}")
                    continue

                if out == "0":
                    logger.info(f"Individual finished without error {ind_idx}: {out}")
                    # individual finished without error
                    self.running_individuals.remove(ind_idx)
                    self.finished_individuals.append(ind_idx)
                    sorted_exit_codes[ind_idx] = 0
                    # set worker to idle
                    self.idle_workers[w_id] = self.running_workers.pop(w_id)

                
                #TODO error control of problematic optimizees
                elif out == "1": 
                    logger.info(f"Individual finished with error {ind_idx}: {out}. Restarting.")
                    self.running_individuals.remove(ind_idx)
                    sorted_exit_codes[ind_idx] = 1
                    # set worker to idle
                    self.idle_workers[w_id] = self.running_workers.pop(w_id)
                    # restart individual 
                    self.restart_individual(gen, ind_idx)
                
                if status_code == None:
                    # Indivdual still running
                    continue

                elif status_code == 0:
                    # Process closed
                    self.running_workers.pop(w_id)
                    logger.info(f"Finished worker {w_id}: {status_code}")
                else: 
                    logger.info(f"Error status worker {w_id}: {status_code}")
                    # worker raised error
                    # TODO depending on what kind of error restart failed worker
                    if status_code > 128 and retry<20:#Error spawning step, wait a bit?
                        logger.info(f"Restarting {w_id} from error {status_code}\n retry {retry}")
                        time.sleep(4)
                        self.restart_worker( w_id)
                        retry += 1
                    else:
                        logger.error("Worker could not be initialized")
                        raise NotImplementedError("Restart failed for worker")


            # assing pending inds to free workers
            self.populate_free_workers(gen=gen)

            if not self.running_individuals and not self.pending_individuals:
                # all individuals finished
                break
            sys.stdout.flush()
            time.sleep(5)
        
        #sorted_exit_codes = [self.finished_individuals[idx].poll() for idx in range(n_inds)]
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
                                "op_trajectory_")
        respath = os.path.join(self.work_paths['results'],
                               "results_")
        f = open(os.path.join(self.path, "run_optimizee.py"), "w")
        f.write('import pickle\n' +
                'import sys\n' +
                'import gc\n' +
                'import os\n' +
                'import logging\n' +
                'import socket\n' +
                'import time\n' +
                'worker_id = sys.argv[1]\n' +
                'logfilename = f"'+self.work_paths['individual_logs']+'/workers_{worker_id}.wlog"\n' +
                'logging.basicConfig(filename=logfilename, filemode="a", level=logging.INFO)\n' +
                'logger = logging.getLogger("Optimizee")\n'+
                'logger.info(socket.gethostname())\n' +
                'outputpipename = f"'+self.work_paths['individual_logs']+'/outputpipe_{worker_id}"\n'+
                'outputpipe = open(outputpipename, "wb")\n' +
                'inputpipename = f"'+self.work_paths['individual_logs']+'/inputpipe_{worker_id}"\n'+
                'inputpipe = open(inputpipename, "r")\n' +
                'running = 1\n' + 
                'while running:\n' +
                '    try:\n' +
                '        logger.info(f"Receiving")\n' +
                '        params = ""\n' +
                '        while not params:\n' +
                '            params = inputpipe.readline()\n' +
                '            time.sleep(5)\n' +
                '        logger.info(f"Params received: {params}")\n' +
                '        params = params.split()\n' +
                '        logger.info(params)\n' +
                '        generation = params[0]\n' +
                '        idx = params[1]\n' +
                '        running = int(params[2])\n' +
                '        if not running:\n' +
                '            break\n' +
                '        handle_trajectory = open("' + trajpath + '"+ str(generation) + ".bin", "rb")\n' +
                '        trajectory = pickle.load(handle_trajectory)\n' +
                '        handle_trajectory.close()\n' +
                '        handle_optimizee = open("' + self.optimizeepath + '", "rb")\n' +
                '        optimizee = pickle.load(handle_optimizee)\n' +
                '        handle_optimizee.close()\n\n' +
                '        logger.info("Trajectory access")\n' +
                '        logger.info(trajectory.individuals)\n' +
                '        logger.info(len(trajectory.individuals[int(generation)]))\n' +
                '        trajectory.individual = trajectory.individuals[int(generation)][int(idx)] \n'+
                '        res = optimizee.simulate(trajectory)\n\n' +
                '        handle_res = open("' + respath + '"+ str(generation) + "_" + str(idx) + ".bin", "wb")\n' +
                '        pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)\n' +
                '        handle_res.close()\n' + 
                '        del optimizee\n' +
                '        outputpipe.write(f"0\\n".encode(\'ascii\'))\n' +
                #'        outputpipe.write("0\\n")\n' +
                '        outputpipe.flush()\n' +
                '        logger.info(f"Finished {idx}")\n' +
                '        gc.collect()\n' +
                '    except Exception as e:\n' +
                '        logger.info(str(e))\n' +
                '        logger.info(f"Params received in except: {params}")\n' +
                '        if params == "":\n' +
                '            continue\n' +
                '        #else:\n' +
                '        #    sys.stderr.write(b"1")\n' +
                '        #    sys.stderr.flush()\n' +
                'outputpipe.close()\n'+
                'inputpipe.close()')
        f.close()

    def dump_traj(self, trajectory):
        """Dumpes trajectory files.
        :param trajectory, object to be dumped"""
        #trajfname = "trajectory_%s.bin" % (trajectory.individual.generation)
        #handle = open(os.path.join(self.work_paths["trajectories"], trajfname),
        #                    "wb")
        #pickle.dump(trajectory, handle, pickle.HIGHEST_PROTOCOL)
        #handle.close()

        mtrajfname = os.path.join(self.work_paths["trajectories"], "trajectory_%s.bin" % (trajectory.individual.generation))
        mtraj = trajectory
        #if os.path.isfile(mtrajfname):
        #    with open(mtrajfname, 'rb') as mhandle:
        #        mtraj = pickle.load(mhandle)
        #        mtraj.individuals[trajectory.individual.generation] = trajectory.individuals[trajectory.individual.generation]

        with open(mtrajfname, 'wb') as mhandle:
            pickle.dump(mtraj, mhandle, pickle.HIGHEST_PROTOCOL)

        #tmpgen = trajectory.individuals[trajectory.individual.generation]
        tmptraj = Trajectory()
        tmptraj.individual = trajectory.individual
        tmptraj.individuals[trajectory.individual.generation] = trajectory.individuals[trajectory.individual.generation]#tmpgen
        trajfname = "op_trajectory_%s.bin" % (trajectory.individual.generation)
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
                        # Deleting the content of the log files after zipping
                        #os.remove(add)
                        f = open(add,'w')
                        f.close()

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
