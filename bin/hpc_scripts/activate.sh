#!/bin/bash/env bashasd

module load CUDA Python SciPy-Stack numba JUBE PyCUDA

export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/TVB/tvb_library
# for sdict which is at this location
export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/L2L/l2l/utils
export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/L2L

export PYTHONPATH=$PYTHONPATH:$PROJECT/$USER/

