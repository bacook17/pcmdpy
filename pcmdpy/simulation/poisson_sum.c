#include <curand_kernel.h>

extern "C"
{
  __global__ void poisson_sum(curandState *global_state, const float *exp_nums, const float *fluxes, const float multiplier, const int num_bands, const int num_bins, const int Nim, float *pixels)
  {
    /* Initialize variables */
    int id_imx = blockIdx.x*blockDim.x + threadIdx.x;
    int id_imy = blockIdx.y*blockDim.y + threadIdx.y;
    int id_pix = (id_imx) + Nim*id_imy;
    /*
    int id_within_block = threadIdx.x + (blockDim.x * threadIdx.y);
    int block_id = blockIdx.x*gridDim.y + blockIdx.y;
    
    int seed_id = id_within_block + ((blockDim.x * blockDim.y) * (block_id % num_procs));
    */
    
    curandState local_state;
    /* If you want to work with 11+ filters, need to update this!! */
    float results[10] = {0.0};
    
    float flux;
    int count;
    
    if ((id_imx < Nim) && (id_imy < Nim)) {
      local_state = global_state[id_pix];
      for (int i = 0; i < num_bins; i++){
	count = curand_poisson(&local_state, exp_nums[i] * multiplier);
	for (int f = 0; f < num_bands; f++){
	  flux = fluxes[i + (f*num_bins)];
	  results[f] += count * flux;
	}
      }
      /* Save results for each band */
      for (int f = 0; f < num_bands; f++){
	pixels[id_pix + (Nim*Nim)*f] = results[f];
      }
      /* Save back state. Important in case we re-draw stars behind */
      global_state[id_pix] = local_state;
    }
    
  }

  __global__ void prepare(curandStateXORWOW *s, const int N,
			     unsigned int *v, const unsigned int o)
  {
    const int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id < N) {
      curand_init(v[id], id, o, &s[id]);
    }
  }
}

