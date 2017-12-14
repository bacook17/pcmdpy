default:
	$(MAKE) pcmdpy_gpu || $(MAKE) pcmdpy_cpu

pcmdpy_gpu:
	pip install .[GPU] --process-dependency-links

pcmdpy_cpu:
	pip install . --process-dependency-links

