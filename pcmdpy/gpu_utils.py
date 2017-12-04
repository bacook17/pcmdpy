import numpy as np
import warnings, os
import multiprocessing
#from reikna import cluda
#from reikna.fft import FFT

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
_code = """
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

def initialize_gpu(n=None):
    """
    This function makes pycuda use GPU number n in the system. If no n is provided, will use the current
    multiprocessing process number
    """
    assert(_GPU_AVAIL)
    if n is None:
        n = multiprocessing.current_process()._identity[0] - 1
        print('for process id: %d'%n)
    else:
        print('using given n: %d'%n)
    
    os.environ['CUDA_DEVICE'] = '%d'%n
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
        global _mod
        print('Starting SourceModule Code')
        _mod = SourceModule(_code, keep=False, no_extern_c=True)
        print('Getting function')
        global _func
        _func = _mod.get_function('poisson_sum')
        print('Past the SourceModule code')
    except:
        print('Something Failed')
    else:
        global _CUDAC_AVAIL
        _CUDAC_AVAIL = True
        print('CUDAC Available')

def draw_image(expected_nums, fluxes, N_scale, gpu=_GPU_ACTIVE, cudac=_CUDAC_AVAIL, fixed_seed=False, **kwargs):
    if gpu:
        if cudac:
            func = _draw_image_cudac
        else:
            func = _draw_image_pycuda
    else:
        func = _draw_image_numpy
    return func(expected_nums, fluxes, N_scale, fixed_seed=fixed_seed, **kwargs)

def seed_getter_fixed(N, value=None):
    assert(_GPU_AVAIL & _GPU_ACTIVE)
    result = pycuda.gpuarray.empty([N], np.int32)
    if value is None:
        #This will draw the same number every time
        np.random.seed(0)
        return pycuda.gpuarray.to_gpu(np.random.randint(0, 2**31 - 1, N).astype(np.int32))
    else:
        return result.fill(value)
        
def _draw_image_cudac(expected_nums, fluxes, N_scale, fixed_seed=False, tolerance=0, d_block=_MAX_2D_BLOCK_DIM, skip_n=1, my_shuffle=False, **kwargs):
    assert(_GPU_AVAIL & _GPU_ACTIVE)
    assert(_CUDAC_AVAIL)

    assert(len(expected_nums) == fluxes.shape[1])

    """
    upper_lim = tolerance**-2.
    use_poisson = (expected_nums <= upper_lim)
    use_fixed = ~use_poisson

    #total flux from all "fully populated" bins above upper_lim
    fixed_fluxes = np.sum(expected_nums[use_fixed]*fluxes[:,use_fixed], axis=1)

    #remove fixed bins, and set proper byte size for cuda
    expected_nums = expected_nums[use_poisson].astype(np.float32)
    fluxes = fluxes[:,use_poisson].astype(np.int32)
    """
    
    expected_nums = expected_nums.astype(np.float32)
    fluxes = fluxes.astype(np.float32)

    N_scale = np.int32(N_scale)

    N_bins = np.int32(len(expected_nums))
    N_bands = np.int32(fluxes.shape[0])
    skip_n = np.int32(skip_n)
    
    if fixed_seed:
        seed_getter = seed_getter_fixed
    else:
        seed_getter = curandom.seed_getter_uniform

    generator = curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
    num_procs = np.int32(generator.block_count)
    result = np.zeros((N_bands, N_scale, N_scale), dtype=np.float32)
    
    block_dim = (d_block, d_block,1)
    grid_dim = (N_scale/d_block + 1, N_scale/d_block + 1)
    _func(generator._state, cuda.In(expected_nums), cuda.In(fluxes), N_bands, N_bins, N_scale,
              cuda.Out(result), skip_n, num_procs, block=block_dim, grid=grid_dim)

    #Add on flux from fully-populated bins
    #result = np.array([result[i] + fixed_fluxes[i] for i in range(N_bands)]).astype(float)
    return result

def _draw_image_pycuda(expected_nums, fluxes, N_scale, fixed_seed=False, tolerance=-1., **kwargs):
    assert(_GPU_AVAIL & _GPU_ACTIVE)

    N_bins = len(expected_nums)
    N_bands = fluxes.shape[0]
    assert(N_bins == fluxes.shape[1])
    if (tolerance < 0.):
        upper_lim = np.inf
    else:
        upper_lim = tolerance**-2.
    
    if fixed_seed:
        seed_getter = seed_getter_fixed
    else:
        seed_getter = curandom.seed_getter_uniform

    generator = pycuda.curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)
    result = np.zeros((N_bands, N_scale*N_scale), dtype=float)

    #Draw stars and cumulate flux for each mass bin
    for b in np.arange(N_bins):
        n_expected = expected_nums[b]
        counts = fluxes[:,b]
        if (n_expected <= upper_lim):
            n_stars = generator.gen_poisson(N_scale*N_scale, np.uint32, n_expected).get()
        #assume no poisson variance
        else:
            n_stars = n_expected
        result += np.array([c * n_stars for c in counts])

    return result.reshape([N_bands, N_scale, N_scale])

def _draw_image_numpy(expected_nums, fluxes, N_scale, fixed_seed=False, tolerance=-1., **kwargs):
    N_bins = len(expected_nums)
    assert(N_bins == fluxes.shape[1])
    if (tolerance < 0.):
        upper_lim = np.inf
    else:
        upper_lim = tolerance**-2.
    if fixed_seed:
        np.random.seed(0)

    realiz_num = np.zeros((N_scale, N_scale, N_bins))
    if not np.isinf(upper_lim):
        realiz_num = np.random.poisson(lam=expected_nums, size=(N_scale, N_scale, N_bins))
    else:
        use_poisson = (expected_nums <= upper_lim)
        use_fixed = ~use_poisson #Assume no poisson variance
        num_poisson = np.sum(use_poisson)
    
        realiz_num[:,:,use_fixed] = expected_nums[use_fixed]
        realiz_num[:,:,use_poisson] = np.random.poisson(lam=expected_nums[use_poisson], size=(N_scale, N_scale, num_poisson))
        
    return np.dot(realiz_num, fluxes.T).T

def gpu_log10(array_in, verbose=False, **kwargs):
    if _GPU_AVAIL:
        return pycuda.cumath.log10(pycuda.gpuarray.to_gpu(array_in)).get()
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
#            assert(len(self.image_shape) == self.ndim)
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
#        #shift = 1 - (np.array(result.shape) / 2)
#        #for i in range(image.ndim):
#        #    result = np.roll(result, shift[i], axis=i)
#        #Remove exterior border
#        if newshape is not None:
#            start = (np.array(result.shape) - np.array(newshape)) / 2
#            end = start + np.array(newshape)
#            myslice = tuple([slice(start[k], end[k]) for k in range(len(newshape))])
#            result = result[myslice]
#        return result
#
#    @classmethod
#    def inflate_sizes(cls, shape1, shape2):
#        try:
#            assert(len(shape1) == len(shape2))
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
#        assert(a.ndim == len(new_shape))
#        assert(a.shape <= new_shape)
#        s_temp = np.array(a.shape)
#        return np.pad(a, [(0, new_shape[i] - s_temp[i]) for i in range(a.ndim)], 'constant')
#
#    def close(self):
#        self.thread.release()
