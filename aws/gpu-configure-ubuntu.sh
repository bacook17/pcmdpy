#!/bin/bash
# Install ecs-init, start docker, and install nvidia-docker
#sudo yum install -y ecs-init
sudo apt-get install rpm

#More details on the below: https://docs.docker.com/engine/installation/linux/docker-ce/ubuntu/
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt-get install \
     apt-transport-https \
     ca-certificates \
     curl \
     software-properties-common
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo apt-key fingerprint 0EBFCD88
sudo add-apt-repository \
        "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
sudo apt-get update
sudo apt-get install docker-ce
sudo docker run hello-world

#More details on the below: http://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-agent-install.html

sudo docker version
echo '!!!!!!! Docker version should be > 1.5.0 !!!!!!!!'


sudo sh -c "echo 'net.ipv4.conf.all.route_localnet = 1' >> /etc/sysctl.conf"
sudo sysctl -p /etc/sysctl.conf

sudo iptables -t nat -A PREROUTING -p tcp -d 169.254.170.2 --dport 80 -j DNAT --to-destination 127.0.0.1:51679
sudo iptables -t nat -A OUTPUT -d 169.254.170.2 -p tcp -m tcp --dport 80 -j REDIRECT --to-ports 51679

sudo sh -c 'iptables-save > /etc/iptables/rules.v4'
sudo mkdir -p /etc/ecs && sudo touch /etc/ecs/ecs.config

sudo echo "ECS_DATADIR=/data
ECS_ENABLE_TASK_IAM_ROLE=true
ECS_ENABLE_TASK_IAM_ROLE_NETWORK_HOST=true
ECS_LOGFILE=/log/ecs-agent.log
ECS_AVAILABLE_LOGGING_DRIVERS=[\"json-file\",\"awslogs\"]
ECS_LOGLEVEL=info
ECS_CLUSTER=default" > /etc/ecs/ecs.config

sudo docker run --name ecs-agent \
     --detach=true \
     --restart=on-failure:10 \
     --volume=/var/run:/var/run \
     --volume=/var/log/ecs/:/log \
     --volume=/var/lib/ecs/data:/data \
     --volume=/etc/ecs:/etc/ecs \
     --net=host \
     --env-file=/etc/ecs/ecs.config \
     amazon/amazon-ecs-agent:latest

sudo service docker start
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
sudo docker pull nvidia/cuda:9.0-cudnn7-devel
COMMAND="sudo nvidia-docker run nvidia/cuda:9.0-cudnn7-devel nvidia-smi"
for i in {1..5}; do $COMMAND && break || sleep 15; done

# Create symlink to latest nvidia-driver version
nvidia_base=/var/lib/nvidia-docker/volumes/nvidia_driver
sudo ln -s $nvidia_base/$(ls $nvidia_base | sort -n  | tail -1) $nvidia_base/latest
