import pickle
import sys
import gc
import os
import logging
import socket
import time
try:
    from mpi4py import MPI
except ImportError:
    mpi_run = False
else:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    mpi_run = True

worker_id = sys.argv[1]
logfilename = f"{logpath}/workers_{worker_id}.wlog"
logging.basicConfig(filename=logfilename, filemode="a", level=logging.INFO)
logger = logging.getLogger("Optimizee")
logger.info(socket.gethostname())
if (mpi_run and rank == 0) or (not mpi_run):
    outputpipename = f"{logpath}/outputpipe_{worker_id}"
    outputpipe = open(outputpipename, "wb")
    inputpipename = f"{logpath}/inputpipe_{worker_id}"
    inputpipe = open(inputpipename, "r")
running = 1
while running:
    try:
        params = ""
        if (mpi_run and rank == 0) or (not mpi_run):
            logger.info(f"Receiving")
            while not params:
                params = inputpipe.readline()
                time.sleep(5)
            logger.info(f"Params received: {params}")
        if mpi_run:
            params = comm.bcast(params,root=0) #
        params = params.split()
        logger.info(params)
        generation = params[0]
        idx = params[1]
        running = int(params[2])
        if not running:
            break
        handle_trajectory = open(f"{trajpath}{generation}.bin", "rb")
        trajectory = pickle.load(handle_trajectory)
        handle_trajectory.close()
        handle_optimizee = open(f"{optimizeepath}", "rb")
        optimizee = pickle.load(handle_optimizee)
        handle_optimizee.close()

        logger.info("Trajectory access")
        logger.info(trajectory.individuals)
        logger.info(len(trajectory.individuals[int(generation)]))
        trajectory.individual = trajectory.individuals[int(generation)][int(idx)]
        res = optimizee.simulate(trajectory)

        if (mpi_run and rank == 0) or (not mpi_run):
            handle_res = open(f"{respath}{generation}_{idx}.bin", "wb")
            pickle.dump(res, handle_res, pickle.HIGHEST_PROTOCOL)
            handle_res.close()
            outputpipe.write(f"0\n".encode('ascii'))
            outputpipe.flush()
            logger.info(f"Finished {idx}")
        del optimizee
        gc.collect()
    except Exception as e:
        if (mpi_run and rank == 0) or (not mpi_run):
            logger.info(str(e))
            logger.info(f"Params received in except: {params}")
        if params == "":
            continue
        #else:
        #    sys.stderr.write(b"1")
        #    sys.stderr.flush()
if (mpi_run and rank == 0) or (not mpi_run):
    outputpipe.close()
    inputpipe.close()
