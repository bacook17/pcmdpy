default: pcmdpy

pcmdpy: 
	$(MAKE) pcmdpy_gpu || $(MAKE) pcmdpy_cpu

pcmdpy_gpu:
	pip install .[GPU]

pcmdpy_cpu:
	pip install .

