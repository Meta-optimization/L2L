from __future__ import print_function

import logging
import itertools
import argparse
import pickle

from tvb.simulator.lab import *
from tvb.basic.logger.builder import get_logger

import os.path
import numpy as np
try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
except ImportError:
	logging.warning('pycuda not available, rateML driver not usable.')

import matplotlib.pyplot as plt

import time
import tqdm

# import sys
# np.set_printoptions(threshold=sys.maxsize)

here = os.path.dirname(os.path.abspath(__file__))
headerhere = here
print("here", here)

# PROJECT = os.getenv('PROJECT')
# USER = os.getenv('USER')
# os.chdir(PROJECT+USER+"/L2L/l2l/optimizees/pse_multi/")
print("wp modeldriver", os.getcwd())

class Driver_Setup:

	def __init__(self):
		self.args = self.parse_args()

		self.logger = get_logger('tvb.rateML')
		self.logger.setLevel(level='INFO' if self.args.verbose else 'WARNING')

		self.checkargbounds()

		self.dt = self.args.delta_time
		self.connectivity = self.tvb_connectivity(self.args.n_regions)
		# self.weights = self.connectivity.weights
		self.weights = self.connectivity.weights / (np.sum(self.connectivity.weights, axis=0) + 1e-12)

		# dump it for visuals
		# np.savetxt("SCweights.csv", self.connectivity.weights, delimiter=",")
		# np.savetxt("SClengths.csv", self.connectivity.tract_lengths, delimiter=",")

		self.lengths = self.connectivity.tract_lengths
		self.tavg_period = 0.1
		self.n_inner_steps = int(self.tavg_period / self.dt)

		self.params = self.setup_params(
		# self.args.n_sweep_arg0,
		# self.args.n_sweep_arg1,
		# self.args.n_sweep_arg2,
		# self.args.n_sweep_arg3,
		# self.args.n_sweep_arg4,
		)

		# bufferlength is based on the minimum of the first swept parameter (speed for many tvb models)
		self.n_work_items, self.n_params = self.params.shape
		self.buf_len_ = ((self.lengths / self.args.speeds_min / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of

		self.states = self.args.states
		self.exposures = self.args.exposures

		if self.args.gpu_info:
			self.logger.setLevel(level='INFO')
			self.gpu_device_info()
			exit(1)

		self.logdata()

	def logdata(self):

		self.logger.info('dt %f', self.dt)
		self.logger.info('s0 %f', self.args.n_sweep_arg0)
		self.logger.info('s1 %f', self.args.n_sweep_arg1)
		# self.logger.info('s2 %f', self.args.n_sweep_arg2)
		# self.logger.info('s3 %f', self.args.n_sweep_arg3)
		# self.logger.info('s4 %f', self.args.n_sweep_arg4)
		self.logger.info('n_nodes %d', self.args.n_regions)
		self.logger.info('weights.shape %s', self.weights.shape)
		self.logger.info('lengths.shape %s', self.lengths.shape)
		self.logger.info('tavg period %s', self.tavg_period)
		self.logger.info('n_inner_steps %s', self.n_inner_steps)
		self.logger.info('params shape %s', self.params.shape)

		self.logger.info('nstep %d', self.args.n_time)
		self.logger.info('n_inner_steps %f', self.n_inner_steps)

		# self.logger.info('single connectome, %d x %d parameter space', self.args.n_sweep_arg0, self.args.n_sweep_arg1)
		self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
		self.logger.info('number of states %d', self.states)
		self.logger.info('model %s', self.args.model)
		self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
		self.logger.info('memory for states array on GPU %d MiB',
						 (self.buf_len * self.n_work_items * self.states * self.args.n_regions * 4) / 1024 ** 2)


	def checkargbounds(self):

		try:
			assert self.args.n_sweep_arg0 > 0, "Min value for [N_SWEEP_ARG0] is 1"
			assert self.args.n_time > 0, "Minimum number for [-n N_TIME] is 1"
			assert self.args.n_regions > 0, "Min value for  [-tvbn n_regions] for default data set is 68"
			assert self.args.blockszx > 0 and self.args.blockszx <= 32,	"Bounds for [-bx BLOCKSZX] are 0 < value <= 32"
			assert self.args.blockszy > 0 and self.args.blockszy <= 32, "Bounds for [-by BLOCKSZY] are 0 < value <= 32"
			assert self.args.delta_time > 0.0, "Min value for [-dt delta_time] is > 0.0, default is 0.1"
			assert self.args.speeds_min > 0.0, "Min value for [-sm speeds_min] is > 0.0, default is 3e-3"
			assert self.args.exposures > 0, "Min value for [-x exposures] is 1"
			assert self.args.states > 0, "Min value for [-s states] is 1"
		except AssertionError as e:
			self.logger.error('%s', e)
			raise

	def tvb_connectivity(self, tvbnodes):
		# white_matter = connectivity.Connectivity.from_file(source_file="connectivity_"+str(tvbnodes)+".zip")
		# white_matter = connectivity.Connectivity.from_file(source_file="paupau.zip")
		white_matter = connectivity.Connectivity.from_file(source_file= here + "/connectivity_zerlaut_68.zip")
		white_matter.configure()
		return white_matter

	def parse_args(self):  # {{{
		parser = argparse.ArgumentParser(description='Run parameter sweep.')

		# for every parameter that needs to be swept, the size can be set
		parser.add_argument('-s0', '--n_sweep_arg0', default=4, help='num grid points for 1st parameter', type=int)
		parser.add_argument('-s1', '--n_sweep_arg1', default=4, help='num grid points for 2st parameter', type=int)
		# parser.add_argument('-s2', '--n_sweep_arg2', default=4, help='num grid points for 3st parameter', type=int)
		# parser.add_argument('-s3', '--n_sweep_arg3', default=4, help='num grid points for 4st parameter', type=int)
		# parser.add_argument('-s4', '--n_sweep_arg4', default=4, help='num grid points for 5st parameter', type=int)
		parser.add_argument('-n', '--n_time', default=400, help='number of time steps to do', type=int)
		parser.add_argument('-v', '--verbose', default=False, help='increase logging verbosity', action='store_true')
		parser.add_argument('-m', '--model', default='zerlaut_func', help="neural mass model to be used during the simulation")
		parser.add_argument('-s', '--states', default=8, type=int, help="number of states for model")
		parser.add_argument('-x', '--exposures', default=2, type=int, help="number of exposures for model")
		parser.add_argument('-l', '--lineinfo', default=False, help='generate line-number information for device code.', action='store_true')
		parser.add_argument('-bx', '--blockszx', default=32, type=int, help="gpu block size x")
		parser.add_argument('-by', '--blockszy', default=32, type=int, help="gpu block size y")
		parser.add_argument('-val', '--validate', default=False, help="enable validation with refmodels", action='store_true')
		parser.add_argument('-r', '--n_regions', default="68", type=int, help="number of tvb nodes")
		parser.add_argument('-p', '--plot_data', type=int, help="plot res data for selected state")
		parser.add_argument('-w', '--write_data', default=False, help="write output data to file: 'tavg_data", action='store_true')
		parser.add_argument('-g', '--gpu_info', default=False, help="show gpu info", action='store_true')
		parser.add_argument('-dt', '--delta_time', default=0.1, type=float, help="dt for simulation")
		parser.add_argument('-sm', '--speeds_min', default=1, type=float, help="min speed for temporal buffer")
		parser.add_argument('--procid', default="0", type=int, help="Number of L2L processes(Only when in L2L)")

		args = parser.parse_args()
		return args

	# Validatoin params
	# S = 0.4
	# b_e = 120.0
	# E_L_e = -64
	# E_L_i = -64
	# T = 19
	def setup_params(self,
		 # n0,
		 # n1,
		 # n2,
		 # n3,
		 # n4,
		 ):
		'''
        This code generates the parameters ranges that need to be set
        '''
		# sweeparam0 = np.linspace(0.2, 0.2, n0)
		# s0 = np.linspace(-64, -64, self.args.n_sweep_arg0)
		# s1 = np.linspace(-64, -64, self.args.n_sweep_arg1)
		# s2 = np.linspace(-64, -64, self.args.n_sweep_arg2)
		# s3 = np.linspace(19, 19, self.args.n_sweep_arg3)
		#
		# # original sweeps
		# # sweeparam0 = np.linspace(0, 0.5, n0)
		# # sweeparam1 = np.linspace(0, 120, n1)
		# # sweeparam2 = np.linspace(-80, -60, n2)
		# # sweeparam3 = np.linspace(-80, -60, n3)
		# # sweeparam4 = np.linspace(5, 40, n4)
		#
		# params = itertools.product(s0, s1, s2, s3)
		# params = itertools.product(s0, s1)
		# params = np.array([vals for vals in params], np.float32)

		# unpickle file from L2L
		paramsfile = open(f'rateml/sweepars_{self.args.procid}', 'rb')
		params = pickle.load(paramsfile)
		paramsfile.close()

		print('paramsvar', np.var(params, axis=0))

		print('paramsonTVB', params.shape)
		print('params', params)
		print('paramsnan?', np.where(np.isnan(params)))

		return params


	def gpu_device_info(self):
		'''
		Get GPU device information
		TODO use this information to give user GPU setting suggestions
		'''
		dev = drv.Device(0)
		print('\n')
		self.logger.info('GPU = %s', dev.name())
		self.logger.info('TOTAL AVAIL MEMORY: %d MiB', dev.total_memory()/1024/1024)

		# get device information
		att = {'MAX_THREADS_PER_BLOCK': [],
			   'MAX_BLOCK_DIM_X': [],
			   'MAX_BLOCK_DIM_Y': [],
			   'MAX_BLOCK_DIM_Z': [],
			   'MAX_GRID_DIM_X': [],
			   'MAX_GRID_DIM_Y': [],
			   'MAX_GRID_DIM_Z': [],
			   'TOTAL_CONSTANT_MEMORY': [],
			   'WARP_SIZE': [],
			   # 'MAX_PITCH': [],
			   'CLOCK_RATE': [],
			   'TEXTURE_ALIGNMENT': [],
			   # 'GPU_OVERLAP': [],
			   'MULTIPROCESSOR_COUNT': [],
			   'SHARED_MEMORY_PER_BLOCK': [],
			   'MAX_SHARED_MEMORY_PER_BLOCK': [],
			   'REGISTERS_PER_BLOCK': [],
			   'MAX_REGISTERS_PER_BLOCK': []}

		for key in att:
			getstring = 'drv.device_attribute.' + key
			# att[key].append(eval(getstring))
			self.logger.info(key + ': %s', dev.get_attribute(eval(getstring)))

class Driver_Execute(Driver_Setup):

	def __init__(self, ds):
		self.args = ds.args
		self.set_CUDAmodel_dir()
		self.weights, self.lengths, self.params = ds.weights, ds.lengths, ds.params
		self.buf_len, self.states, self.n_work_items = ds.buf_len, ds.states, ds.n_work_items
		self.n_inner_steps, self.n_params, self.dt = ds.n_inner_steps, ds.n_params, ds.dt
		self.exposures, self.logger = ds.exposures, ds.logger
		self.connectivity = ds.connectivity

	def set_CUDAmodel_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "../generatedModels", self.args.model.lower() + '.c')

	def set_CUDA_ref_model_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "../generatedModels/cuda_refs", self.args.model.lower() + '.c')

	def compare_with_ref(self, tavg0):
		self.args.model = self.args.model + 'ref'
		self.set_CUDA_ref_model_dir()
		tavg1 = self.run_simulation()

		# compare output to check if same as template
		# comparison = (tavg0.ravel() == tavg1.ravel())
		# self.logger.info('Templated version is similar to original %d:', comparison.all())
		self.logger.info('Corr.coef. of model with %s is: %f', self.args.model,
						 np.corrcoef(tavg0.ravel(), tavg1.ravel())[0, 1])

		return np.corrcoef(tavg0.ravel(), tavg1.ravel())[0, 1]

	def make_kernel(self, source_file, warp_size, args, lineinfo=True, nh='nh'):

		try:
			with open(source_file, 'r') as fd:
				source = fd.read()
				source = source.replace('M_PI_F', '%ff' % (np.pi, ))
				opts = ['--ptxas-options=-v', '-maxrregcount=32']
				# if lineinfo:
				opts.append('-lineinfo')
				opts.append('-g')
				opts.append('-DWARP_SIZE=%d' % (warp_size, ))
				opts.append('-DNH=%s' % (nh, ))

				idirs = [here, headerhere]
				self.logger.info('nvcc options %r', opts)

				try:
					network_module = SourceModule(
							source, options=opts, include_dirs=idirs,
							no_extern_c=True,
							keep=False,)
				except drv.CompileError as e:
					self.logger.error('Compilation failure \n %s', e)
					exit(1)

				# generic func signature creation
				# _Z7zerlautjjjjjfPfS_S_S_S_
				# _Z11bold_updateifPfS_S_
				# mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model, 'jjjjjfPfS_S_S_S_')
				mod_func = '_Z7zerlautjjjjjfPfS_S_S_S_'

				step_fn = network_module.get_function(mod_func)

			with open('rateml/covar.c', 'r') as fd:
				source = fd.read()
				opts = ['-ftz=true']  # for faster rsqrtf in corr
				opts.append('-DWARP_SIZE=%d' % (warp_size,))
				# opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x,))
				covar_module = SourceModule(source, options=opts)
				covar_fn = covar_module.get_function('update_cov')
				cov_corr_fn = covar_module.get_function('cov_to_corr')

		except FileNotFoundError as e:
			self.logger.error('%s.\n  Generated model filename should match model on cmdline', e)
			exit(1)

		return step_fn, covar_fn, cov_corr_fn

	def cf(self, array):#{{{
		# coerce possibly mixed-stride, double precision array to C-order single precision
		return array.astype(dtype='f', order='C', copy=True)#}}}

	def nbytes(self, data):#{{{
		# count total bytes used in all data arrays
		nbytes = 0
		for name, array in data.items():
			nbytes += array.nbytes
		return nbytes#}}}

	def make_gpu_data(self, data):#{{{
		# put data onto gpu
		gpu_data = {}
		for name, array in data.items():
			try:
				gpu_data[name] = gpuarray.to_gpu(self.cf(array))
			except drv.MemoryError as e:
				self.gpu_mem_info()
				self.logger.error(
					'%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU',
					e, self.params.size)
				exit(1)
		return gpu_data#}}}

	def release_gpumem(self, gpu_data):
		for name, array in gpu_data.items():
			try:
				gpu_data[name].gpudata.free()
			except drv.MemoryError as e:
				self.logger.error('%s.\n\t Freeing mem error', e)
				exit(1)

	def gpu_mem_info(self):

		cmd = "nvidia-smi -q -d MEMORY"#,UTILIZATION"
		os.system(cmd)  # returns the exit code in unix

	def run_simulation(self):

		# setup data#{{{
		data = { 'weights': self.weights, 'lengths': self.lengths, 'params': self.params.T }
		base_shape = self.n_work_items,
		for name, shape in dict(
			tavg0=(self.exposures, self.args.n_regions,),
			tavg1=(self.exposures, self.args.n_regions,),
			state=(self.buf_len, self.states * self.args.n_regions),
			covar_means=(2 * self.args.n_regions,),
			covar_cov=(self.args.n_regions, self.args.n_regions,),
			corr=(self.args.n_regions, self.args.n_regions,),
			).items():
			# memory error exception for compute device
			try:
				data[name] = np.zeros(shape + base_shape, 'f')
			except MemoryError as e:
				self.logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large '
							 'for this compute device',
							 e, self.args.n_sweep_arg0, self.args.n_sweep_arg1)
				exit(1)

		gpu_data = self.make_gpu_data(data)#{{{

		# setup CUDA stuff#{{{
		step_fn, covar_fn, cov_corr_fn = self.make_kernel(
			# source_file=self.args.filename,
			source_file=here + '/zerlaut.c',
			warp_size=32,
			# block_dim_x=self.args.n_sweep_arg0,
			# ext_options=preproccesor_defines,
			# caching=args.caching,
			args=self.args,
			lineinfo=self.args.lineinfo,
			nh=self.buf_len,
			)#}}}

		# setup simulation#{{{
		tic = time.time()

		n_streams = 32
		streams = [drv.Stream() for i in range(n_streams)]
		events = [drv.Event() for i in range(n_streams)]
		tavg_unpinned = []

		try:
			tavg = drv.pagelocked_zeros((n_streams,) + data['tavg0'].shape, dtype=np.float32)
		except drv.MemoryError as e:
			self.logger.error(
				'%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU',
				e, self.params.size)
			exit(1)

		# determine optimal grid recursively
		def dog(fgd):
			maxgd, mingd = max(fgd), min(fgd)
			maxpos = fgd.index(max(fgd))
			if (maxgd - 1) * mingd * bx * by >= nwi:
				fgd[maxpos] = fgd[maxpos] - 1
				dog(fgd)
			else:
				return fgd

		# n_sweep_arg0 scales griddim.x, n_sweep_arg1 scales griddim.y
		# form an optimal grid recursively
		bx, by = self.args.blockszx, self.args.blockszy
		nwi = self.n_work_items
		rootnwi = int(np.ceil(np.sqrt(nwi)))
		gridx = int(np.ceil(rootnwi / bx))
		gridy = int(np.ceil(rootnwi / by))

		final_block_dim = bx, by, 1

		fgd = [gridx, gridy]
		dog(fgd)
		final_grid_dim = fgd[0], fgd[1]
		# final_grid_dim = 4, 4

		assert gridx * gridy * bx * by >= nwi

		self.logger.info('work items %r', self.n_work_items)
		self.logger.info('history shape %r', gpu_data['state'].shape)
		self.logger.info('gpu_data %s', gpu_data['tavg0'].shape)
		self.logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))
		self.logger.info('final block dim %r', final_block_dim)
		self.logger.info('final grid dim %r', final_grid_dim)

		# run simulation#{{{
		nstep = self.args.n_time

		self.gpu_mem_info() if self.args.verbose else None

		try:
			for i in tqdm.trange(nstep, file=sys.stdout):

				try:
					event = events[i % n_streams]
					stream = streams[i % n_streams]

					if i > 0:
						stream.wait_for_event(events[(i - 1) % n_streams])

					step_fn(np.uintc(i * self.n_inner_steps), np.uintc(self.args.n_regions), np.uintc(self.buf_len),
							np.uintc(self.n_inner_steps), np.uintc(self.n_work_items), np.float32(self.dt),
							gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
							gpu_data['tavg%d' % (i%2,)],
							block=final_block_dim, grid=final_grid_dim)

					event.record(streams[i % n_streams])
				except drv.LaunchError as e:
					self.logger.error('%s', e)
					exit(1)

				# print('gpudatashape', gpu_data['tavg%d' % (i%2,)].shape)

				tavgk = 'tavg%d' % ((i + 1) % 2,)

				# if i >= (nstep // 2):
				# 	i_time = i - nstep // 2
				# 	# update_cov (covar_cov is output, tavgk and covar_means are input)
				# 	covar_fn(np.uintc(i_time), np.uintc(self.args.n_regions), np.uintc(self.n_work_items),
				# 			 gpu_data['covar_cov'], gpu_data['covar_means'], gpu_data[tavgk],
				# 			 # gpu_data['corr'], gpu_data['covar_means'], gpu_data[tavgk],
				# 			 block=final_block_dim, grid=final_grid_dim,
				# 			 stream=stream)

				# async wrt. other streams & host, but not this stream.
				if i >= n_streams:
					stream.synchronize()
					tavg_unpinned.append(tavg[i % n_streams].copy())

				drv.memcpy_dtoh_async(tavg[i % n_streams], gpu_data[tavgk].ptr, stream=stream)

				# if i == (nstep - 1):
				# # cov_to_corr(covar_cov is input, and corr output)
				# 	cov_corr_fn(np.uintc(nstep // 2), np.uintc(self.args.n_regions), np.uintc(self.n_work_items),
				# 				gpu_data['covar_cov'], gpu_data['corr'],
				# 				# block=(couplings.size, 1, 1), grid=(speeds.size, 1), stream=stream)
				# 				block=final_block_dim, grid=final_grid_dim,
				# 				stream=stream)

			# recover uncopied data from pinned buffer
			if nstep > n_streams:
				for i in range(nstep % n_streams, n_streams):
					stream.synchronize()
					tavg_unpinned.append(tavg[i].copy())

			for i in range(nstep % n_streams):
				stream.synchronize()
				tavg_unpinned.append(tavg[i].copy())

			corr = gpu_data['corr'].get()

		except drv.LogicError as e:
			self.logger.error('%s. Check the number of states of the model or '
						 'GPU block shape settings blockdim.x/y %r, griddim %r.',
						 e, final_block_dim, final_grid_dim)
			exit(1)
		except drv.RuntimeError as e:
			self.logger.error('%s', e)
			exit(1)


		# self.logger.info('kernel finish..')
		# release pinned memory
		tavg = np.array(tavg_unpinned)

		# also release gpu_data
		self.release_gpumem(gpu_data)

		self.logger.info('kernel finished')
		return tavg, corr

	def plot_output(self, tavg):
		plt.plot((tavg[:, self.args.plot_data, :, 0]), 'k', alpha=.2)
		plt.show()

	def write_output(self, tavg, cut_transient):
		from datetime import datetime
		# timestring = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

		filename = '/tavg_data'
		tavg_file = open(here + filename, 'wb')
		pickle.dump(tavg[cut_transient:, :, :, :], tavg_file)
		tavg_file.close()

	def write_output_corr(self, corr):
		from datetime import datetime
		# timestring = datetime.now().strftime("%d.%m.%Y-%H:%M:%S")

		filename = '/corr'
		corr_file = open(here + filename, 'wb')
		pickle.dump(corr, corr_file)
		corr_file.close()

	def calc_corrcoef(self, corr):
		# calculate correlation between SC and simulated FC. SC is the weights of TVB simulation.
		SC = self.connectivity.weights / self.connectivity.weights.max()
		ccFCSC = np.zeros(self.n_work_items, 'f')
		for i in range(self.n_work_items):
			ccFCSC[i] = np.corrcoef(corr[:, :, i].ravel(), SC.ravel())[0, 1]

		return ccFCSC

	def calc_corrcoef_FC(self, corr):
		# calculate correlation between simulated FC from Goldman simulation for specific set of params for Ex
		# and Inhibitory firing rates. 68 nodes need be the case
		# shape of the Goldman FC
		pearsonfile = open('rateml/pearson_0.4_72_-64_-64_19', 'rb')
		FCExIn = pickle.load(pearsonfile)
		pearsonfile.close()

		SC = self.connectivity.weights / self.connectivity.weights.max()

		# print("SCshapee", SC.shape)
		# print("corrshape", corr.shape)
		# print("FCXIshape", FCExIn.shape[:][:][0])

		ccFCFC = np.zeros(self.n_work_items, 'f')
		for i in range(self.n_work_items):
			ccFCFC[i] = np.corrcoef(
				corr[:,:,i].ravel(),
				FCExIn[:,:,0].ravel())[0, 1]

		# print(FCExIn[:,:,0])

		return ccFCFC

	def calc_corrcoef_TAVG(self, tavg, cut_transient):
		# calc pearon with numpy
		tavgFC = np.zeros((68, 68, self.n_work_items), 'f')
		for i in range(self.n_work_items):
			tavgFC[:, :, i] = np.corrcoef(np.transpose(tavg[cut_transient:, 0, :, i]) * 1e3)

		return tavgFC


	def run_all(self):

		np.random.seed(79)

		tic = time.time()

		tavg0, corr = self.run_simulation()
		toc = time.time()
		elapsed = toc - tic

		if (self.args.validate == True):
			self.compare_with_ref(tavg0)

		cut_transient = 2000

		self.plot_output(tavg0) if self.args.plot_data is not None else None
		self.write_output(tavg0, cut_transient) if self.args.write_data else None
		self.write_output_corr(corr) if self.args.write_data else None
		self.logger.info('Output shape (simsteps, states, bnodes, n_params) %s', tavg0.shape)
		self.logger.info('Finished CUDA simulation successfully in: {0:.3f}'.format(elapsed))
		self.logger.info('and in {0:.3f} M step/s'.format(
			1e-6 * self.args.n_time * self.n_inner_steps * self.n_work_items / elapsed))

		# cut transient relative to sim time
		# cut_transient = 2 * self.args.n_time // 5
		cut_transient = 2000

		tavgFC = self.calc_corrcoef_TAVG(tavg0, cut_transient)

		# the functional structural correlation computation
		# self.logger.info('tavg0 %s', tavg0)
		# self.logger.info('corr %s', corr.shape)
		# self.logger.info('corr %s', corr)
		# ccFCSC = self.calc_corrcoef(corr)
		ccFCFC = self.calc_corrcoef_FC(tavgFC)
		self.logger.info('tavgFC %s', tavgFC.shape)
		# self.logger.info('tavgFC %s', tavgFC)
		self.logger.info('fitness shape %s', ccFCFC.shape)
		self.logger.info('max fitness %s', np.max(ccFCFC))
		resL2L_file = open(here + f'/result_{self.args.procid}', 'wb')
		pickle.dump(ccFCFC, resL2L_file)
		# pickle.dump(self.calc_corrcoef(corr), resL2L_file)
		resL2L_file.close()

		# where are the nans
		print('tavgnan?', np.where(np.isnan(tavg0)))
		print('tavgFCnan?', np.where(np.isnan(tavgFC)))
		print('fcfcnan?', np.where(np.isnan(ccFCFC)))

		return


if __name__ == '__main__':

	driver_setup = Driver_Setup()
	Driver_Execute(driver_setup).run_all()
