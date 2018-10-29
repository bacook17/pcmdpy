#include <curand_kernel.h>

extern "C"
{
  __global__ void poisson_sum(curandState *global_state, const float *exp_nums, const float *fluxes, const float multiplier, const int num_bands, const int num_bins, const int N, float *pixels, const int skip_n, const int num_procs)
  {
    /* Initialize variables */
    int id_imx = blockIdx.x*blockDim.x + threadIdx.x;
    int id_imy = blockIdx.y*blockDim.y + threadIdx.y;
    int id_pix = (id_imx) + N*id_imy;
    int id_within_block = threadIdx.x + (blockDim.x * threadIdx.y);
    int block_id = blockIdx.x*gridDim.y + blockIdx.y;
    
    int seed_id = id_within_block + ((blockDim.x * blockDim.y) * (block_id % num_procs));
    
    curandState local_state = global_state[seed_id];
    float results[10] = {0.0};
    
    float flux;
    int count, skip;
    
    if ((id_imx < N) && (id_imy < N)) {
      /* Update local_state, to make sure values are very random */
      skip = skip_n * block_id;
      skipahead(skip, &local_state);
      for (int i = 0; i < num_bins; i++){
	count = curand_poisson(&local_state, exp_nums[i] * multiplier);
	for (int f = 0; f < num_bands; f++){
	  flux = fluxes[i + (f*num_bins)];
	  results[f] += count * flux;
	}
      }
      /* Save results for each band */
      for (int f = 0; f < num_bands; f++){
	pixels[id_pix + (N*N)*f] = results[f];
      }
    }
    
    /* Save back state */
    global_state[seed_id] = local_state;
  }
}
