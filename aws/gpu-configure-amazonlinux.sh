#!/bin/bash
# Install ecs-init, start docker, and install nvidia-docker
sudo yum install -y ecs-init
sudo service docker start

#Start the ecs-init upstart job.
#sudo cp ecs.config /etc/ecs/ecs.config
#sudo start ecs
wget https://github.com/NVIDIA/nvidia-docker/releases/download/v1.0.1/nvidia-docker-1.0.1-1.x86_64.rpm
sudo rpm -ivh --nodeps nvidia-docker-1.0.1-1.x86_64.rpm

# Validate installation
rpm -ql nvidia-docker
rm nvidia-docker-1.0.1-1.x86_64.rpm

# Make sure the NVIDIA kernel modules and driver files are bootstraped
# Otherwise running a GPU job inside a container will fail with "cuda: unknown exception"
echo '#!/bin/bash' | sudo tee /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe > /dev/null
echo 'nvidia-modprobe -u -c=0' | sudo tee --append /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe > /dev/null
sudo chmod +x /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe
sudo /var/lib/cloud/scripts/per-boot/00_nvidia-modprobe

# Start the nvidia-docker-plugin and run a container with
# nvidia-docker (retry up to 4 times if it fails initially)
sudo -b nohup nvidia-docker-plugin > /tmp/nvidia-docker.log
sudo docker pull bacook17/pcmdpy_gpu
COMMAND="sudo nvidia-docker run bacook17/pcmdpy_gpu nvidia-smi"
for i in {1..5}; do $COMMAND && break || sleep 5; done

# Create symlink to latest nvidia-driver version
nvidia_base=/var/lib/nvidia-docker/volumes/nvidia_driver
sudo ln -s $nvidia_base/$(ls $nvidia_base | sort -n  | tail -1) $nvidia_base/latest

