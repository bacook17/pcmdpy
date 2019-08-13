import numpy as np
import warnings
import os
import multiprocessing
from pkg_resources import resource_filename
from sys import stderr

global _GPU_AVAIL, _GPU_ACTIVE, _MAX_THREADS_PER_BLOCK, _MAX_2D_BLOCK_DIM
_GPU_AVAIL = False
_GPU_ACTIVE = False
_MAX_THREADS_PER_BLOCK = None
_MAX_2D_BLOCK_DIM = None

try:
    import pycuda
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule
    from pycuda import curandom
    from pycuda import cumath

except ImportError as e:
    mess = e.__str__()  # error message
    _GPU_FAIL_REASON = ""
    if 'No module named \'pycuda\'' in mess:
        _GPU_FAIL_REASON = (
            'pycuda not installed.\nPlease ensure you are using a machine with an '
            'NVidia GPU and CUDA installed, then install pycuda\n(such as via '
            ' "pip install pycuda")')
    elif 'libcuda' in mess:
        _GPU_FAIL_REASON = (
            'libcuda not found.\nPlease ensure you are using a machine with an '
            'NVidia GPU and CUDA installed.\nSee https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html'
        )
    else:
        _GPU_FAIL_REASON = mess
    stderr.flush()
else:
    _GPU_AVAIL = True

_MAX_THREADS_PER_BLOCK = 1024
_MAX_2D_BLOCK_DIM = 32


def initialize_gpu(n=0):
    """
    This function makes pycuda use GPU number n in the system. If no n is provided, will use the current
    multiprocessing process number
    """
    assert _GPU_AVAIL, (
        "Requested GPU acceleration unavailable, for reason:\n   {}".format(_GPU_FAIL_REASON)
    )
    if n is None:
        n = multiprocessing.current_process()._identity[0] - 1
        print(('for process id: {0:d}'.format(n)))
    
    src_file = resource_filename('pcmdpy', 'simulation/') + 'poisson_sum.c'

    with open(src_file, 'r') as f:
        src_code = f.read()

    os.environ['CUDA_DEVICE'] = '{0:d}'.format(n)
    try:
        import pycuda.autoinit
    except ImportError as e:
        raise ImportError(
            "Requested GPU acceleration unavailable, for reason:\n"
            "   failed to autoinitialize pycuda")

    try:
        _MAX_THREADS_PER_BLOCK = pycuda.autoinit.device.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        _MAX_2D_BLOCK_DIM = int(np.floor(np.sqrt(_MAX_THREADS_PER_BLOCK)))
    except:
        warnings.warn('Reverting to default MAX_THREADS_PER_BLOCK '
                      'and MAX_2D_BLOCK_DIM', RuntimeWarning)
        _MAX_THREADS_PER_BLOCK = 1024
        _MAX_2D_BLOCK_DIM = 32
        _STATE_SIZE = 48
    
    try:
        module = SourceModule(src_code, keep=False, no_extern_c=True)
        global poisson_sum_gpu
        poisson_sum_gpu = module.get_function('poisson_sum')
        global prepare_states
        prepare_states = module.get_function('prepare')
        prepare_states.prepare('PiPi')  # arguments are pointer, int, pointer, int
    except cuda.CompileError as e:
        raise RuntimeError(
            'Something failed in compiling C source.\n'
            '{}'.format(e.msg))
    global _GPU_ACTIVE
    _GPU_ACTIVE = True
    return _GPU_ACTIVE


class XORWOWStatesArray(object):

    def __init__(self, N, data=None, fixed_seed=True, offset=0):
        assert _GPU_ACTIVE
        self.N = N
        self.state_size = pycuda.characterize.sizeof("curandStateXORWOW",
                                                     "#include <curand_kernel.h>")
        self.size = self.N * self.state_size
        self.gpudata = cuda.mem_alloc(self.size)
        if data is None:
            self.set_states(fixed_seed=fixed_seed, offset=offset)
        else:
            if not isinstance(data, cuda.DeviceAllocation):
                raise ValueError("Data must be a valid Pycuda Device Allocation")
            cuda.memcpy_dtod(self.gpudata, data, self.size)

    def __len__(self):
        return self.N

    def set_states(self, fixed_seed=True, offset=0):
        if fixed_seed:
            np.random.seed(0)
        else:
            np.random.seed()
        seed = pycuda.gpuarray.to_gpu(
            np.random.randint(0, 2**31 - 1, self.N).astype(np.int32))
        d_block = _MAX_THREADS_PER_BLOCK
        grid_dim = (int(self.N//d_block + 1), 1, 1)
        block_dim = (int(d_block), 1, 1)
        prepare_states.prepared_call(grid_dim, block_dim, self.gpudata, self.N,
                                     seed.gpudata, offset)
        seed.gpudata.free()

    def copy(self):
        return XORWOWStatesArray(self.N, data=self.gpudata)

    def free(self):
        self.gpudata.free()

    def __del__(self):
        self.free()


def draw_image(*args, gpu=_GPU_ACTIVE, **kwargs):
    if gpu:
        func = _draw_image_gpu
    else:
        func = _draw_image_numpy
    return func(*args, **kwargs)


def seed_getter_fixed(N, value=None):
    assert _GPU_ACTIVE, (
        "Can\'t use GPU implementation: _GPU_ACTIVE set to False, "
        "likely because initialization not run or failed")
    result = pycuda.gpuarray.empty([N], np.int32)
    if value is None:
        # This will draw the same number every time
        np.random.seed(0)
        return pycuda.gpuarray.to_gpu(np.random.randint(0, 2**31 - 1, N).astype(np.int32))
    else:
        return result.fill(value)


def _draw_image_gpu(expected_nums, fluxes, Nim, filters, dust_frac,
                    dust_mean, dust_std, d_states, fixed_seed=False, tolerance=0,
                    fudge_mag=0.0,
                    d_block=_MAX_2D_BLOCK_DIM, **kwargs):
    assert _GPU_ACTIVE, (
        "Can\'t use GPU implementation: _GPU_ACTIVE set to False, "
        "likely because initialization not run or failed")

    assert (len(expected_nums) == fluxes.shape[1]), (
        "expected_nums must have same shape as fluxes")

    expected_nums = expected_nums.astype(np.float32)
    fluxes = fluxes.astype(np.float32)

    N_bins = len(expected_nums)
    N_bands = fluxes.shape[0]

    d_expected_nums = cuda.In(expected_nums)
    d_fluxes = cuda.In(fluxes)

    result_front = np.zeros((N_bands, Nim, Nim), dtype=np.float32)
    result_behind = np.zeros((N_bands, Nim, Nim), dtype=np.float32)
    
    block_dim = (int(d_block), int(d_block), 1)
    grid_dim = (int(Nim//d_block + 1), int(Nim//d_block + 1))

    Npix_fudge_factor = 10.**(fudge_mag) - 1.0
    
    # draw stars behind dust screen
    poisson_sum_gpu(d_states, d_expected_nums, d_fluxes, np.float32(dust_frac),
                    np.int32(N_bands), np.int32(N_bins), np.int32(Nim),
                    np.float32(Npix_fudge_factor),
                    cuda.Out(result_behind), block=block_dim, grid=grid_dim)

    # draw stars in front of dust screen
    if dust_frac <= 0.99:
        poisson_sum_gpu(d_states, d_expected_nums, d_fluxes,
                        np.float32(1. - dust_frac), np.int32(N_bands),
                        np.int32(N_bins), np.int32(Nim),
                        np.float32(Npix_fudge_factor),
                        cuda.Out(result_front), block=block_dim, grid=grid_dim)

    if fixed_seed:
        np.random.seed(0)
    else:
        np.random.seed()
    dust_screen = np.random.lognormal(mean=dust_mean, sigma=dust_std,
                                      size=(Nim, Nim))
    reddening = np.array([10.**(-0.4 * dust_screen * f.red_per_ebv)
                          for f in filters])
    return result_front + result_behind*reddening


def _draw_image_numpy(expected_nums, fluxes, Nim, filters, dust_frac,
                      dust_mean, dust_std, fudge_mag=0.0, d_states=None, fixed_seed=False, tolerance=-1., **kwargs):
    N_bins = len(expected_nums)
    assert (N_bins == fluxes.shape[1]), (
        "fluxes.shape[1] should match number of bins")
    if (tolerance < 0.):
        upper_lim = np.inf
    else:
        upper_lim = tolerance**-2.
    if fixed_seed:
        np.random.seed(0)
    else:
        np.random.seed()

    realiz_front = np.zeros((Nim, Nim, N_bins))
    realiz_behind = np.zeros((Nim, Nim, N_bins))
    if fudge_mag >= 1e-5:
        fudge_Npix = 10.**(0.4*fudge_mag)
        random_Npix = np.random.uniform(1.0, fudge_Npix, size=(Nim, Nim, 1))
        expected_nums = (expected_nums * random_Npix)
        assert expected_nums.shape == (Nim, Nim, N_bins)
    if np.isinf(upper_lim):
        realiz_front = np.random.poisson(lam=expected_nums*(1. - dust_frac), size=(Nim, Nim, N_bins))
        realiz_behind = np.random.poisson(lam=expected_nums*dust_frac, size=(Nim, Nim, N_bins))
    else:
        use_poisson = (expected_nums <= upper_lim)
        use_fixed = ~use_poisson #Assume no poisson variance
        num_poisson = np.sum(use_poisson)
    
        realiz_front[:, :, use_fixed] = expected_nums[use_fixed] * (1. - dust_frac)
        realiz_behind[:, :, use_fixed] = expected_nums[use_fixed] * dust_frac
        realiz_front[:,:,use_poisson] = np.random.poisson(lam=expected_nums[use_poisson]*(1. - dust_frac), size=(Nim, Nim, num_poisson))
        realiz_behind[:,:,use_poisson] = np.random.poisson(lam=expected_nums[use_poisson]*dust_frac, size=(Nim, Nim, num_poisson))

    result_front = np.dot(realiz_front, fluxes.T).T
    result_behind = np.dot(realiz_behind, fluxes.T).T

    dust_screen = np.random.lognormal(mean=dust_mean, sigma=dust_std, size=(Nim, Nim))
    reddening = np.array([10.**(-0.4 * dust_screen * f.red_per_ebv) for f in filters])
    
    return result_front + result_behind*reddening


def gpu_log10(array_in, verbose=False, **kwargs):
    if _GPU_ACTIVE:
        if type(array_in) is not np.ndarray:
            array_in = np.array(array_in)
        if len(array_in) <= 1e6:
            return np.log10(array_in)
        else:
            return cumath.log10(pycuda.gpuarray.to_gpu(array_in)).get()
    else:
        if verbose:
            warnings.warn('gpu_log10 using cpu, because gpu not available.', RuntimeWarning)
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
