#include <stdio.h>
#ifdef TEST_COVAR
#include <math.h>
#endif

// stable one-pass co-moment algo, cf wikipedia

#ifndef TEST_COVAR
__global__
#endif
void update_cov(
    unsigned int i_sample,
    unsigned int n_node,
    unsigned int nwi,
    float * __restrict__ cov,
    float * __restrict__ means,
    const float * __restrict__ data
)
{
//#ifdef TEST_COVAR
//    const unsigned int it = 0;
//    const unsigned int nt = 1;
//#else
//    const unsigned int it = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//    const unsigned int nt = blockDim.x * gridDim.x * gridDim.y;
//#endif
    // index for 2d grid with 2d blocks
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const unsigned int it = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

//    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x; old global index
    const unsigned int size = nwi;

    if (it >= size) return;

    if (i_sample == 0)
    {
        for (int i_node = 0; i_node < n_node; i_node++){
            means[i_node * size + it] = data[i_node * size + it];
//            printf("data[%d] = %f\n", i_node * size + it, means[i_node * size + it]);
        }
        return;
    }

    const float recip_n = 1.0f / i_sample;
    // double buffer to avoid copying memory
    // set starting address?
    float *next_mean = means, *prev_mean = means;
    if ((i_sample % 2) == 0) {
	    prev_mean += n_node * size;
//	    printf("prevmean %p\n", prev_mean);
    } else {
	    next_mean += n_node * size;
//	    printf("nextmean %p\n", next_mean);
    }

    for (int i_node = 0; i_node < n_node; i_node++)
//    for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node += blockDim.y)
    {
        if (i_node >= n_node) continue;

        int i_idx = i_node * size + it;
        next_mean[i_idx] = prev_mean[i_idx] + (data[i_idx] - prev_mean[i_idx]) * recip_n;
//        printf("next_mean[%d] = %f\n", i_idx, next_mean[i_idx]);

    }
    
    // TODO shared mem useful here?
    for (int i_node = 0; i_node < n_node; i_node++)
//    for (unsigned int i_node = threadIdx.y; i_node < n_node; i_node += blockDim.y)
    {
        if (i_node >= n_node) continue;

        int i_idx = i_node * size + it;
        float data_mean_i = data[i_idx] - prev_mean[i_idx];
//        printf("data[%d] = %f\n", i_idx, data[i_idx]);
//        printf("prev_mean[%d] = %f\n", i_idx, prev_mean[i_idx]);
//        printf("data_mean_i@%d = %f\n", i_idx, data_mean_i);

        for (int j_node = 0; j_node < n_node; ++j_node)
        {
            int j_idx = j_node * size + it;
            float data_mean_j = data[j_idx] - next_mean[j_idx];
//            printf("data_mean_j@%d = %f\n", j_idx, data_mean_j);
            int cij_idx = (j_node * n_node + i_node) * size + it;
            cov[cij_idx] += data_mean_j * data_mean_i;
        }
    }
}

#ifndef TEST_COVAR
__global__
#endif
void cov_to_corr(
    unsigned int n_sample,
    unsigned int n_node,
    unsigned int nwi,
    float * __restrict__ cov,
    float * __restrict__ corr
)
{
//#ifdef TEST_COVAR
//    const unsigned int it = 0;
//    const unsigned int nt = 1;
//#else
//    const unsigned int it = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
//    const unsigned int nt = blockDim.x * gridDim.x * gridDim.y;
//#endif

    // global index for 2d grid with 2d blocks
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    const unsigned int it = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

//    const unsigned int it = (gridDim.x * blockDim.x * threadIdx.y) + threadIdx.x; old global index
    const unsigned int size = nwi;

    if (it >= size) return;

    float recip_n_samp = 1.0f / n_sample;

    // normalize comoment to covariance
    for (unsigned int ij = 0; ij < (n_node * n_node); ++ij)
	    cov[ij*size + it] *= recip_n_samp;

    // compute correlation coefficient
#define COV(i, j) cov[((i)*n_node + (j))*size + it]
#define CORR(i, j) corr[((i)*n_node + (j))*size + it]

    for (int i = 0; i < n_node; i++)
//    for (unsigned int i = threadIdx.y; i < n_node; i += blockDim.y)
    {
//        if (i >= n_node) continue;

        float var_i = COV(i, i);
        for (unsigned int j = 0; j < n_node; ++j)
        {
            float var_j = COV(j, j);
            CORR(i, j) = COV(i, j) * rsqrtf(var_i * var_j);
        }
    }
}
