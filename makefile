default: gpu

gpu: pcmdpy_gpu

cpu: pcmdpy

develop: 
	@echo "---------------------------------------"
	@echo "installing pcmdpy only, NO dependencies"
	python -m pip install -e . --no-deps
	@echo "successfully completed installing pcmdpy with NO dependencies"
	@echo "---------------------------------------"

pcmdpy:
	@echo "---------------------------------------"
	@echo "installing pcmdpy with CPU support only"
	python -m pip install . --user --upgrade
	@echo "successfully completed installing pcmdpy with CPU support only"
	@echo "---------------------------------------"

pcmdpy_gpu: pcmdpy
	@echo "---------------------------------------"
	@echo "attempting to install GPU support for pcmdpy"
	@python -m pip install -q .[GPU] --user --upgrade || (echo "GPU support not available. Confirm NVIDIA GPU is available, the NVIDIA driver and CUDA are installed."; exit 1)
	@echo "successfully completed installing pcmdpy with GPU support"
	@echo "---------------------------------------"
