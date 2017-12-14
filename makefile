code = pcmdpy/driver.py pcmdpy/fit_model.py pcmdpy/galaxy.py pcmdpy/gpu_utils.py pcmdpy/instrument.py pcmdpy/isochrones.py pcmdpy/priors.py pcmdpy/utils.py

default: pcmdpy

pcmdpy: $(code)
	$(MAKE) pcmdpy_gpu || $(MAKE) pcmdpy_cpu

pcmdpy_gpu:
	pip install .[GPU]

pcmdpy_cpu:
	pip install .

