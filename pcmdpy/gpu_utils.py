import numpy as np
import warnings
import os
import multiprocessing
from pcmdpy.utils import my_assert
# from reikna import cluda
# from reikna.fft import FFT

try:
    import pycuda
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda import curandom
    from pycuda import cumath

except ImportError as e:
    print('GPU acceleration not available, sorry')
    _GPU_AVAIL = False
    mess = e.__str__() #error message
    if 'No module named pycuda' in mess:
        warnings.warn('pycuda not installed.',ImportWarning)
        print('pycuda not installed.')
    elif 'libcuda' in mess:
        warnings.warn('libcuda not found, likely because no GPU available.', RuntimeWarning)
        print('libcuda not found, likely because no GPU available.')
    else:
        warnings.warn(mess, ImportWarning)
        print(mess)
else:
    _GPU_AVAIL = True
    print('GPU acceleration enabled')

_GPU_ACTIVE = False
_CUDAC_AVAIL = False
_MAX_THREADS_PER_BLOCK = 1024
_MAX_2D_BLOCK_DIM = 32

_single_code = """
   #include <curand_kernel.h>

   extern "C"
   {
   __global__ void poisson_sum(curandState *global_state, const float *exp_nums, const float *fluxes, const int num_bands, const int num_bins, const int N, float *pixels, const int skip_n, const int num_procs)
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
             count = curand_poisson(&local_state, exp_nums[i]);
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
"""

_code = """
   #include <curand_kernel.h>
   #include <math.h>
   extern "C"
   {
   __global__ void poisson_sum(curandState *global_state, const float *exp_nums, const float *fluxes, const float *red_per_ebv, float dust_frac, float dust_mean, float dust_sig, const int num_bands,
                               const int num_bins, const int N, float *pixels, const int skip_n, const int num_procs)
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
      int count_front, count_behind, skip;
      float dust;
      float reddening;

      if ((id_imx < N) && (id_imy < N)) {
          /* Update local_state, to make sure values are very random */
          skip = skip_n * block_id;
          skipahead(skip, &local_state);
          /* Draw dust for this pixel */
          dust = curand_log_normal(&local_state, dust_mean, dust_sig);
          for (int i = 0; i < num_bins; i++){
             /* distribute some starsin front of the dust screen, some behind */
             count_front = curand_poisson(&local_state, exp_nums[i] * (1.0 - dust_frac));
             count_behind = curand_poisson(&local_state, exp_nums[i] * dust_frac);
             for (int f = 0; f < num_bands; f++){
                reddening = powf(10., -0.4 * (dust * red_per_ebv[f]));
                flux = fluxes[i + (f*num_bins)];
                /* add stars in front of dust screen */
                results[f] += count_front * flux;
                /* add stars behind dust screen */
                results[f] += count_behind * flux * reddening;
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


"""


_double_code = """
   #include <curand_kernel.h>
   #include <math.h>
   extern "C"
   {
   __global__ void poisson_sum(curandState *global_state, const float *exp_nums, const float *fluxes, float dust_frac, const int num_bands,
                               const int num_bins, const int N, float *pixels_front, float *pixels_behind, const int skip_n, const int num_procs)
   {
      /* Initialize variables */
      int id_imx = blockIdx.x*blockDim.x + threadIdx.x;
      int id_imy = blockIdx.y*blockDim.y + threadIdx.y;
      int id_pix = (id_imx) + N*id_imy;
      int id_within_block = threadIdx.x + (blockDim.x * threadIdx.y);
      int block_id = blockIdx.x*gridDim.y + blockIdx.y;

      int seed_id = id_within_block + ((blockDim.x * blockDim.y) * (block_id % num_procs));

      curandState local_state = global_state[seed_id];
      float results_front[10] = {0.0};
      float results_behind[10] = {0.0};

      float flux;
      int count_front, count_behind, skip;

      if ((id_imx < N) && (id_imy < N)) {
          /* Update local_state, to make sure values are very random */
          skip = skip_n * block_id;
          skipahead(skip, &local_state);
          for (int i = 0; i < num_bins; i++){
             /* distribute some starsin front of the dust screen, some behind */
             count_front = curand_poisson(&local_state, exp_nums[i] * (1.0 - dust_frac));
             count_behind = curand_poisson(&local_state, exp_nums[i] * dust_frac);
             for (int f = 0; f < num_bands; f++){
                flux = fluxes[i + (f*num_bins)];
                /* add stars in front of dust screen */
                results_front[f] += count_front * flux;
                /* add stars behind dust screen */
                results_behind[f] += count_behind * flux;
             }
          }
          /* Save results for each band */
          for (int f = 0; f < num_bands; f++){
             pixels_front[id_pix + (N*N)*f] = results_front[f];
             pixels_behind[id_pix + (N*N)*f] = results_behind[f];
          }
      }

      /* Save back state */
      global_state[seed_id] = local_state;
   }
   }
"""

def initialize_gpu(n=None):
    """
    This function makes pycuda use GPU number n in the system. If no n is provided, will use the current
    multiprocessing process number
    """
    my_assert(_GPU_AVAIL,
              "Can\'t initialize GPU, _GPU_AVAIL is set to False")
    if n is None:
        n = multiprocessing.current_process()._identity[0] - 1
        print(('for process id: {0:d}'.format(n)))
    else:
        print(('using given n: {0:d}'.format(n)))
    
    os.environ['CUDA_DEVICE'] = '{0:d}'.format(n)
    import pycuda.autoinit

    global _GPU_ACTIVE
    _GPU_ACTIVE = True

    global _MAX_THREADS_PER_BLOCK
    global _MAX_2D_BLOCK_DIM
    try:
        _MAX_THREADS_PER_BLOCK = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        _MAX_2D_BLOCK_DIM = int(np.floor(np.sqrt(_MAX_THREADS_PER_BLOCK)))
    except:
        _MAX_THREADS_PER_BLOCK = 1024
        _MAX_2D_BLOCK_DIM = 32
    
    try:
        global _single_mod
        global _mod
        global _double_mod
        print('Starting SourceModule Code')
        _single_mod = SourceModule(_single_code, keep=False, no_extern_c=True)
        _mod = SourceModule(_code, keep=False, no_extern_c=True)
        _double_mod = SourceModule(_double_code, keep=False, no_extern_c=True)
        print('Getting function')
        global _single_func
        global _func
        global _double_func
        _single_func = _single_mod.get_function('poisson_sum')
        _func = _mod.get_function('poisson_sum')
        _double_func = _double_mod.get_function('poisson_sum')
        print('Past the SourceModule code')
    except cuda.CompileError as e:
        print('Something Failed')
        print(e.msg)
        print(e.stderr)
    else:
        global _CUDAC_AVAIL
        _CUDAC_AVAIL = True
        print('CUDAC Available')

def draw_image(expected_nums, fluxes, N_scale, filters, dust_frac, dust_mean, dust_std,
               gpu=_GPU_ACTIVE, fixed_seed=False, **kwargs):
    if gpu:
        func = _draw_image_cudac
    else:
        func = _draw_image_numpy
    return func(expected_nums, fluxes, N_scale, filters, dust_frac, dust_mean, dust_std, fixed_seed=fixed_seed, **kwargs)

def seed_getter_fixed(N, value=None):
    my_assert(_GPU_AVAIL & _GPU_ACTIVE,
              ("Can\'t use seed_getter_fixed: either _GPU_AVAIL_ or "
               "_GPU_ACTIVE are set to False"))
    result = pycuda.gpuarray.empty([N], np.int32)
    if value is None:
        #This will draw the same number every time
        np.random.seed(0)
        return pycuda.gpuarray.to_gpu(np.random.randint(0, 2**31 - 1, N).astype(np.int32))
    else:
        return result.fill(value)
        
def _draw_image_cudac(expected_nums, fluxes, N_scale, filters, dust_frac,
                      dust_mean, dust_std, fixed_seed=False, tolerance=0, d_block=_MAX_2D_BLOCK_DIM, skip_n=1, 
                      mode='default', **kwargs):
    my_assert(_GPU_AVAIL & _GPU_ACTIVE,
              ("Can\'t use seed_getter_fixed: either _GPU_AVAIL_ or "
               "_GPU_ACTIVE are set to False"))
    my_assert(_CUDAC_AVAIL, ("Trying to use cudac implementation, but "
                             "_CUDAC_AVAIL set to False"))

    my_assert(len(expected_nums) == fluxes.shape[1],
              "expected_nums must have same shape as fluxes")

    expected_nums = expected_nums.astype(np.float32)
    fluxes = fluxes.astype(np.float32)
    red_per_ebvs = np.array([f.red_per_ebv for f in filters]).astype(np.float32)

    N_scale = N_scale

    N_bins = len(expected_nums)
    N_bands = fluxes.shape[0]
    
    if fixed_seed:
        seed_getter = seed_getter_fixed
    else:
        seed_getter = curandom.seed_getter_uniform

    generator = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
    num_procs = generator.block_count
    result_front = np.zeros((N_bands, N_scale, N_scale), dtype=np.float32)
    result_behind = np.zeros((N_bands, N_scale, N_scale), dtype=np.float32)
    
    block_dim = (int(d_block), int(d_block), 1)
    grid_dim = (int(N_scale//d_block + 1), int(N_scale//d_block + 1))
    if mode == 'default':
        _func(generator._state, cuda.In(expected_nums), cuda.In(fluxes),
              cuda.In(red_per_ebvs), np.float32(dust_mean), np.float32(dust_std),
              np.int32(N_bands), np.int32(N_bins), np.int32(N_scale),
              cuda.Out(result_behind), np.in32(skip_n), np.int32(num_procs))
        return result_behind
    else:
        dust_screen = np.random.lognormal(mean=dust_mean, sigma=dust_std,
                                          size=(N_scale, N_scale))
        reddening = np.array([10.**(-0.4 * dust_screen * f.red_per_ebv)
                              for f in filters])
        if mode == 'single':
            _single_func(generator._state, cuda.In(expected_nums), cuda.In(fluxes),
                         np.int32(N_bands), np.int32(N_bins), np.int32(N_scale),
                         cuda.Out(result_behind), np.int32(skip_n), np.int32(num_procs),
                         block=block_dim, grid=grid_dim)
        else:
            _double_func(generator._state, cuda.In(expected_nums),
                         cuda.In(fluxes), np.float32(dust_frac),
                         np.int32(N_bands), np.int32(N_bins), np.int32(N_scale),
                         cuda.Out(result_front), cuda.Out(result_behind),
                         np.int32(skip_n), np.int32(num_procs),
                         block=block_dim, grid=grid_dim)
        return result_front + result_behind*reddening

def _draw_image_numpy(expected_nums, fluxes, N_scale, filters, dust_frac,
                      dust_mean, dust_std, fixed_seed=False, tolerance=-1., **kwargs):
    N_bins = len(expected_nums)
    my_assert(N_bins == fluxes.shape[1],
              "fluxes.shape[1] should match number of bins")
    if (tolerance < 0.):
        upper_lim = np.inf
    else:
        upper_lim = tolerance**-2.
    if fixed_seed:
        np.random.seed(0)

    realiz_front = np.zeros((N_scale, N_scale, N_bins))
    realiz_behind = np.zeros((N_scale, N_scale, N_bins))
    if not np.isinf(upper_lim):
        realiz_front = np.random.poisson(lam=expected_nums*(1. - dust_frac), size=(N_scale, N_scale, N_bins))
        realiz_behind = np.random.poisson(lam=expected_nums*dust_frac, size=(N_scale, N_scale, N_bins))
    else:
        use_poisson = (expected_nums <= upper_lim)
        use_fixed = ~use_poisson #Assume no poisson variance
        num_poisson = np.sum(use_poisson)
    
        realiz_front[:, :, use_fixed] = expected_nums[use_fixed] * (1. - dust_frac)
        realiz_behind[:, :, use_fixed] = expected_nums[use_fixed] * dust_frac
        realiz_front[:,:,use_poisson] = np.random.poisson(lam=expected_nums[use_poisson]*(1. - dust_frac), size=(N_scale, N_scale, num_poisson))
        realiz_behind[:,:,use_poisson] = np.random.poisson(lam=expected_nums[use_poisson]*dust_frac, size=(N_scale, N_scale, num_poisson))

    result_front = np.dot(realiz_front, fluxes.T).T
    result_behind = np.dot(realiz_behind, fluxes.T).T

    dust_screen = np.random.lognormal(mean=dust_mean, sigma=dust_std, size=(N_scale, N_scale))
    reddening = np.array([10.**(-0.4 * dust_screen * f.red_per_ebv) for f in filters])
    
    return result_front + result_behind*reddening

def gpu_log10(array_in, verbose=False, **kwargs):
    if _GPU_AVAIL:
        return cumath.log10(pycuda.gpuarray.to_gpu(array_in)).get()
    else:
        if verbose:
            warnings.warn('gpu_log10 using cpu, because gpu not available.',RuntimeWarning)
        return np.log10(array_in)

#class PSFConvolver():
#
#    def __init__(self, image_shape=None, psf_shape=None):
#        #The GPU thread to submit kernels to
#        self.thread = cluda.cuda_api().Thread.create()
#        if (image_shape is not None) and (psf_shape is not None):
#            self._setup(image_shape, psf_shape)
#        else:
#            self.psf_shape = None
#            self.ndim = None
#            self.fshape = None
#            self.fft = None
#            self.fftc = None
#            self.image_shape = None
#            self.initialized = False
#        
#    def _setup(self, image_shape, psf_shape):
#        self.image_shape = np.array(image_shape)
#        self.psf_shape = np.array(psf_shape)
#        self.ndim = len(self.psf_shape)
#        try:
#            my_assert(len(self.image_shape) == self.ndim)
#        except:
#            raise AssertionError('Input images must have same number of dimensions')
#        
#        #The image and psf will be padded to a larger image that can fit both together
#        #Inflate this padded shape to be a power of 2
#        self.fshape = self.inflate_sizes(self.image_shape, self.psf_shape)
#        self.fslice = tuple([slice(0, self.image_shape[i] + self.psf_shape[i] - 1) for i in range(self.ndim)])
#        psf_temp = pycuda.gpuarray.GPUArray(tuple(self.fshape), np.complex64)
#
#        #The FFT object, initialized to the shape "self.fshape2"
#        self.fft = FFT(psf_temp)
#        self.fftc = self.fft.compile(self.thread)
#
#        self.initialized = True
#
#    def convolve_image(self, image, psf, mode='valid'):
#        padded_shape = self.inflate_sizes(image.shape, psf.shape)
#        if not self.initialized or np.any(padded_shape != self.fshape):
#            print('Reinitializing Convolver for new image sizes')
#            self._setup(image.shape, psf.shape)
#        #Inflate the arrays to the proper, power of 2 shape
#        image_pad = self.pad_end(image, self.fshape).astype(np.complex64)
#        psf_pad = self.pad_end(psf, self.fshape).astype(np.complex64)
#        image_g = pycuda.gpuarray.to_gpu(image_pad)
#        psf_g = pycuda.gpuarray.to_gpu(psf_pad)
#
#        #FFT the image and psf arrays, in place
#        self.fftc(image_g, image_g)
#        self.fftc(psf_g, psf_g)
#        result_g = image_g * psf_g
#        self.fftc(result_g, result_g, inverse=True)
#        result = result_g.get().real[self.fslice]
#
#        if mode == 'full':
#            return self.center(result)
#        elif mode == 'same':
#            return self.center(result, newshape=self.image_shape)
#        elif mode == 'valid':
#            return self.center(result, newshape=(self.image_shape - self.psf_shape + 1))
#
#    @classmethod
#    def center(cls, image, newshape=None):
#        #Center the quandrants of the image
#        result = image.copy()
#        #shift = 1 - (np.array(result.shape) // 2)
#        #for i in range(image.ndim):
#        #    result = np.roll(result, shift[i], axis=i)
#        #Remove exterior border
#        if newshape is not None:
#            start = (np.array(result.shape) - np.array(newshape)) // 2
#            end = start + np.array(newshape)
#            myslice = tuple([slice(start[k], end[k]) for k in range(len(newshape))])
#            result = result[myslice]
#        return result
#
#    @classmethod
#    def inflate_sizes(cls, shape1, shape2):
#        try:
#            my_assert(len(shape1) == len(shape2))
#        except:
#            raise AssertionError('Input images must have same number of dimensions')
#        return np.array([cls.next_power2(shape1[i] + shape2[i] - 1) for i in range(len(shape1))]) 
#
#    @classmethod
#    def next_power2(cls, n):
#        #Return the next power of two larger than (or equal) to n
#        # ex: next_power2(19) -> 32
#        return int(2**np.ceil(np.log2(n)))
#
#    @classmethod
#    def pad_end(cls, a, new_shape):
#        #Pad zeros to the end of each dimension of array "a" until it is shape "new_shape"
#        #In 2D, this results in zeros to the bottom and right of the matrix
#        new_shape = tuple(new_shape)
#        my_assert(a.ndim == len(new_shape))
#        my_assert(a.shape <= new_shape)
#        s_temp = np.array(a.shape)
#        return np.pad(a, [(0, new_shape[i] - s_temp[i]) for i in range(a.ndim)], 'constant')
#
#    def close(self):
#        self.thread.release()
