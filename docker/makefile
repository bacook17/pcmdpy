default: pcmdpy

pcmdpy: 
	docker build -f Dockerfile_pcmdpy -t bacook17/pcmdpy_gpu --no-cache . && docker push bacook17/pcmdpy_gpu

base: Dockerfile_base
	docker build -f Dockerfile_base -t bacook17/gpu_base .

base_hard:
	docker build -f Dockerfile_base -t bacook17/gpu_base --no-cache .

overhaul: base_hard pcmdpy

all: base pcmdpy
