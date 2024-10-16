#!/bin/bash/env bashasd

module load CUDA Python SciPy-Stack numba PyCUDA

export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/test_EOSC/TVB/tvb_library
# for sdict which is at this location
export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/test_EOSC/L2L/l2l/utils
export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/test_EOSC/L2L
