default: gpu

gpu: pcmdpy_gpu

cpu: pcmdpy_cpu

update:
	python setup.py install

pcmdpy_only: 
	@echo "---------------------------------------"
	@echo "installing pcmdpy only, NO dependencies"
	python -m pip install . --user --upgrade --no-deps
	@echo "successfully completed installing pcmdpy with NO dependencies"
	@echo "---------------------------------------"

pcmdpy_cpu:
	@echo "---------------------------------------"
	@echo "installing pcmdpy with CPU support only"
	python -m pip install . --user --upgrade
	@echo "successfully completed installing pcmdpy with CPU support only"
	@echo "---------------------------------------"

pcmdpy_gpu: pcmdpy_cpu
	@echo "---------------------------------------"
	@echo "attempting to install GPU support for pcmdpy"
	@python -m pip install -q .[GPU] --user --upgrade || (echo "GPU support not available. Confirm NVIDIA GPU is available, the NVIDIA driver and CUDA are installed."; exit 1)
	@echo "successfully completed installing pcmdpy with GPU support"
	@echo "---------------------------------------"

manual:
	python setup.py install clean
