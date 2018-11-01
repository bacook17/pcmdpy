default: gpu

gpu: install_gpu

cpu: install

develop:  
	@echo "---------------------------------------"
	@echo "installing pcmdpy only, NO dependencies"
	pip install -e . --no-deps
	@echo "successfully completed installing pcmdpy with NO dependencies"
	@echo "---------------------------------------"

install:
	@echo "---------------------------------------"
	@echo "installing pcmdpy with CPU support only"
	pip install . --upgrade
	@echo "successfully completed installing pcmdpy with CPU support only"
	@echo "---------------------------------------"

install_gpu: install
	@echo "---------------------------------------"
	@echo "attempting to install GPU support for pcmdpy"
	@pip install -q .[GPU] --upgrade || (echo "GPU support not available. Confirm NVIDIA GPU is available, the NVIDIA driver and CUDA are installed."; exit 1)
	@echo "successfully completed installing pcmdpy with GPU support"
	@echo "---------------------------------------"
