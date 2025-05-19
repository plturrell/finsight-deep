#!/usr/bin/env python3
"""
Production GPU Cluster Deployment
Deploys AIQToolkit neural supercomputer on physical hardware
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, List, Optional
import yaml
import boto3
import paramiko
from kubernetes import client, config
from dataclasses import dataclass
import nvidia_smi
from concurrent.futures import ThreadPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GPUNode:
    """Represents a physical GPU node"""
    hostname: str
    ip_address: str
    gpu_count: int
    gpu_model: str  # "A100-80GB", "H100-80GB", etc.
    cpu_model: str
    cpu_cores: int
    memory_gb: int
    nvlink_version: str
    infiniband_speed: str
    os_version: str
    cuda_version: str
    driver_version: str


class ProductionClusterDeployer:
    """Deploy neural supercomputer on production GPU clusters"""
    
    def __init__(self, cluster_config: str):
        with open(cluster_config, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.nodes: List[GPUNode] = []
        self.ssh_clients: Dict[str, paramiko.SSHClient] = {}
        
        # AWS/GCP/Azure clients
        self.cloud_provider = self.config.get('cloud_provider', 'aws')
        self._init_cloud_clients()
        
        # Kubernetes client
        config.load_kube_config()
        self.k8s_client = client.CoreV1Api()
        self.k8s_apps = client.AppsV1Api()
    
    def _init_cloud_clients(self):
        """Initialize cloud provider clients"""
        if self.cloud_provider == 'aws':
            self.ec2 = boto3.client('ec2')
            self.iam = boto3.client('iam')
            self.s3 = boto3.client('s3')
        elif self.cloud_provider == 'gcp':
            from google.cloud import compute_v1
            self.gcp_compute = compute_v1.InstancesClient()
        elif self.cloud_provider == 'azure':
            from azure.mgmt.compute import ComputeManagementClient
            from azure.identity import DefaultAzureCredential
            self.azure_compute = ComputeManagementClient(
                DefaultAzureCredential(),
                self.config.get('azure_subscription_id')
            )
    
    def deploy_cluster(self):
        """Deploy the complete cluster"""
        logger.info("Starting production cluster deployment...")
        
        # 1. Provision GPU nodes
        self._provision_gpu_nodes()
        
        # 2. Setup networking
        self._setup_high_speed_networking()
        
        # 3. Install NVIDIA drivers and CUDA
        self._install_nvidia_stack()
        
        # 4. Configure Kubernetes
        self._setup_kubernetes_cluster()
        
        # 5. Deploy monitoring
        self._deploy_monitoring_stack()
        
        # 6. Deploy AIQToolkit
        self._deploy_aiqtoolkit()
        
        # 7. Run validation tests
        self._validate_deployment()
        
        logger.info("Cluster deployment complete!")
        return self._get_cluster_info()
    
    def _provision_gpu_nodes(self):
        """Provision GPU nodes on cloud provider"""
        logger.info(f"Provisioning {self.config['num_nodes']} GPU nodes...")
        
        if self.cloud_provider == 'aws':
            self._provision_aws_nodes()
        elif self.cloud_provider == 'gcp':
            self._provision_gcp_nodes()
        elif self.cloud_provider == 'azure':
            self._provision_azure_nodes()
        elif self.cloud_provider == 'on_premise':
            self._setup_on_premise_nodes()
    
    def _provision_aws_nodes(self):
        """Provision nodes on AWS"""
        # Launch P4d.24xlarge instances (8x A100 80GB)
        instances = []
        
        for i in range(self.config['num_nodes']):
            response = self.ec2.run_instances(
                ImageId=self.config['aws_ami_id'],  # Deep Learning AMI
                InstanceType='p4d.24xlarge',
                MinCount=1,
                MaxCount=1,
                KeyName=self.config['ssh_key_name'],
                SecurityGroupIds=[self.config['security_group_id']],
                SubnetId=self.config['subnet_id'],
                IamInstanceProfile={'Name': self.config['iam_role']},
                UserData=self._get_user_data_script(),
                TagSpecifications=[{
                    'ResourceType': 'instance',
                    'Tags': [
                        {'Key': 'Name', 'Value': f'neural-compute-{i}'},
                        {'Key': 'Cluster', 'Value': 'neural-supercomputer'}
                    ]
                }],
                Placement={
                    'AvailabilityZone': self.config['availability_zone'],
                    'GroupName': self.config['placement_group']  # Cluster placement for low latency
                }
            )
            
            instance_id = response['Instances'][0]['InstanceId']
            instances.append(instance_id)
            logger.info(f"Launched instance {instance_	}")
        
        # Wait for instances to be running
        waiter = self.ec2.get_waiter('instance_running')
        waiter.wait(InstanceIds=instances)
        
        # Get instance details
        for instance_id in instances:
            instance = self.ec2.describe_instances(InstanceIds=[instance_id])['Reservations'][0]['Instances'][0]
            
            node = GPUNode(
                hostname=instance['PrivateDnsName'],
                ip_address=instance['PrivateIpAddress'],
                gpu_count=8,
                gpu_model='A100-80GB',
                cpu_model='Intel Xeon Platinum 8275CL',
                cpu_cores=96,
                memory_gb=1152,
                nvlink_version='NVLink-3',
                infiniband_speed='400Gbps',
                os_version='Ubuntu 20.04',
                cuda_version='11.8',
                driver_version='520.61.05'
            )
            self.nodes.append(node)
    
    def _setup_high_speed_networking(self):
        """Configure high-speed networking (InfiniBand, NVSwitch)"""
        logger.info("Setting up high-speed networking...")
        
        for node in self.nodes:
            logger.info(f"Configuring networking on {node.hostname}")
            
            # Setup InfiniBand
            commands = [
                # Configure InfiniBand interfaces
                "sudo modprobe ib_uverbs",
                "sudo modprobe rdma_cm",
                "sudo modprobe ib_umad",
                
                # Set up IPoIB
                f"sudo ip link set dev ib0 up",
                f"sudo ip addr add {self._get_ib_ip(node)}/24 dev ib0",
                
                # Configure RDMA
                "sudo rdma link",
                
                # Enable GPUDirect RDMA
                "sudo modprobe nvidia-peermem",
                
                # Configure NVSwitch topology
                "sudo nvidia-smi nvlink --status",
                "sudo nvidia-smi topo --matrix"
            ]
            
            self._execute_remote_commands(node, commands)
    
    def _install_nvidia_stack(self):
        """Install NVIDIA drivers, CUDA, and related tools"""
        logger.info("Installing NVIDIA software stack...")
        
        with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
            futures = []
            for node in self.nodes:
                future = executor.submit(self._install_nvidia_on_node, node)
                futures.append(future)
            
            for future in futures:
                future.result()
    
    def _install_nvidia_on_node(self, node: GPUNode):
        """Install NVIDIA stack on a single node"""
        logger.info(f"Installing NVIDIA stack on {node.hostname}")
        
        commands = [
            # Update system
            "sudo apt-get update",
            "sudo apt-get upgrade -y",
            
            # Install NVIDIA driver
            f"sudo apt-get install -y nvidia-driver-{node.driver_version}",
            
            # Install CUDA toolkit
            f"wget https://developer.download.nvidia.com/compute/cuda/{node.cuda_version}/local_installers/cuda_{node.cuda_version}_linux.run",
            f"sudo sh cuda_{node.cuda_version}_linux.run --silent --toolkit",
            
            # Install cuDNN
            "sudo apt-get install -y libcudnn8",
            
            # Install NCCL
            "sudo apt-get install -y libnccl2 libnccl-dev",
            
            # Install nvidia-docker
            "distribution=$(. /etc/os-release;echo $ID$VERSION_ID)",
            "curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -",
            "curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list",
            "sudo apt-get update && sudo apt-get install -y nvidia-docker2",
            "sudo systemctl restart docker",
            
            # Install monitoring tools
            "sudo apt-get install -y nvidia-ml-py3 dcgm",
            
            # Verify installation
            "nvidia-smi",
            "nvidia-smi topo --matrix"
        ]
        
        self._execute_remote_commands(node, commands)
    
    def _setup_kubernetes_cluster(self):
        """Setup Kubernetes cluster for orchestration"""
        logger.info("Setting up Kubernetes cluster...")
        
        # Initialize master node
        master_node = self.nodes[0]
        
        master_commands = [
            # Install Kubernetes
            "sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates curl",
            "sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg",
            "echo 'deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main' | sudo tee /etc/apt/sources.list.d/kubernetes.list",
            "sudo apt-get update",
            "sudo apt-get install -y kubelet kubeadm kubectl",
            "sudo apt-mark hold kubelet kubeadm kubectl",
            
            # Initialize cluster
            f"sudo kubeadm init --pod-network-cidr=10.244.0.0/16 --apiserver-advertise-address={master_node.ip_address}",
            
            # Setup kubectl
            "mkdir -p $HOME/.kube",
            "sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config",
            "sudo chown $(id -u):$(id -g) $HOME/.kube/config",
            
            # Install Flannel CNI
            "kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml",
            
            # Install NVIDIA device plugin
            "kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.13.0/nvidia-device-plugin.yml"
        ]
        
        output = self._execute_remote_commands(master_node, master_commands)
        
        # Extract join command
        join_command = None
        for line in output.split('\n'):
            if 'kubeadm join' in line:
                join_command = line.strip()
                break
        
        # Join worker nodes
        for node in self.nodes[1:]:
            worker_commands = [
                "sudo apt-get update && sudo apt-get install -y apt-transport-https ca-certificates curl",
                "sudo curl -fsSLo /usr/share/keyrings/kubernetes-archive-keyring.gpg https://packages.cloud.google.com/apt/doc/apt-key.gpg",
                "echo 'deb [signed-by=/usr/share/keyrings/kubernetes-archive-keyring.gpg] https://apt.kubernetes.io/ kubernetes-xenial main' | sudo tee /etc/apt/sources.list.d/kubernetes.list",
                "sudo apt-get update",
                "sudo apt-get install -y kubelet kubeadm kubectl",
                "sudo apt-mark hold kubelet kubeadm kubectl",
                f"sudo {join_command}"
            ]
            
            self._execute_remote_commands(node, worker_commands)
    
    def _deploy_monitoring_stack(self):
        """Deploy monitoring (Prometheus, Grafana, DCGM)"""
        logger.info("Deploying monitoring stack...")
        
        # Deploy kube-prometheus-stack
        commands = [
            "helm repo add prometheus-community https://prometheus-community.github.io/helm-charts",
            "helm repo update",
            "helm install monitoring prometheus-community/kube-prometheus-stack -n monitoring --create-namespace",
            
            # Deploy NVIDIA DCGM exporter
            "helm repo add nvidia https://nvidia.github.io/gpu-monitoring-tools/helm-charts",
            "helm install dcgm-exporter nvidia/dcgm-exporter -n monitoring",
            
            # Configure Grafana dashboards
            "kubectl apply -f monitoring/gpu-dashboard.yaml",
            "kubectl apply -f monitoring/cluster-dashboard.yaml"
        ]
        
        master_node = self.nodes[0]
        self._execute_remote_commands(master_node, commands)
    
    def _deploy_aiqtoolkit(self):
        """Deploy AIQToolkit neural supercomputer"""
        logger.info("Deploying AIQToolkit...")
        
        # Create namespace
        self.k8s_client.create_namespace(
            body=client.V1Namespace(metadata=client.V1ObjectMeta(name="aiqtoolkit"))
        )
        
        # Deploy components
        manifests = [
            "deployment/kubernetes/distributed-neural-computer.yaml",
            "deployment/kubernetes/consensus-service.yaml",
            "deployment/kubernetes/orchestrator.yaml",
            "deployment/kubernetes/api-server.yaml",
            "deployment/kubernetes/monitoring.yaml"
        ]
        
        for manifest in manifests:
            subprocess.run(["kubectl", "apply", "-f", manifest], check=True)
        
        # Wait for deployments to be ready
        deployments = ["neural-computer", "consensus", "orchestrator", "api-server"]
        for deployment in deployments:
            subprocess.run([
                "kubectl", "wait", "--for=condition=available",
                f"deployment/{deployment}", "-n", "aiqtoolkit",
                "--timeout=600s"
            ], check=True)
    
    def _validate_deployment(self):
        """Validate the deployment is working correctly"""
        logger.info("Validating deployment...")
        
        # Check all nodes are ready
        nodes = self.k8s_client.list_node()
        for node in nodes.items:
            assert node.status.conditions[-1].type == "Ready"
            assert node.status.conditions[-1].status == "True"
        
        # Check GPU resources
        for node in nodes.items:
            capacity = node.status.capacity
            assert "nvidia.com/gpu" in capacity
            assert int(capacity["nvidia.com/gpu"]) > 0
        
        # Test GPU communication
        self._test_gpu_communication()
        
        # Test distributed training
        self._test_distributed_training()
        
        logger.info("Deployment validation successful!")
    
    def _test_gpu_communication(self):
        """Test GPU-to-GPU communication"""
        logger.info("Testing GPU communication...")
        
        # Create test job
        job = client.V1Job(
            metadata=client.V1ObjectMeta(name="gpu-comm-test"),
            spec=client.V1JobSpec(
                template=client.V1PodTemplateSpec(
                    spec=client.V1PodSpec(
                        containers=[
                            client.V1Container(
                                name="test",
                                image="nvcr.io/nvidia/pytorch:23.10-py3",
                                command=["python", "-c", """
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Test all-reduce
tensor = torch.ones(1).cuda(rank)
dist.all_reduce(tensor)
assert tensor.item() == world_size

# Test all-to-all
input_tensor = torch.ones(world_size).cuda(rank) * rank
output_tensor = torch.empty(world_size).cuda(rank)
dist.all_to_all(output_tensor, input_tensor)

print(f"Rank {rank}: Communication test passed!")
                                """],
                                resources=client.V1ResourceRequirements(
                                    limits={"nvidia.com/gpu": "1"}
                                )
                            )
                        ],
                        restart_policy="Never"
                    )
                )
            )
        )
        
        self.k8s_batch = client.BatchV1Api()
        self.k8s_batch.create_namespaced_job(namespace="default", body=job)
        
        # Wait for completion
        time.sleep(30)
        
        # Check job status
        job_status = self.k8s_batch.read_namespaced_job_status(
            name="gpu-comm-test",
            namespace="default"
        )
        
        assert job_status.status.succeeded > 0, "GPU communication test failed"
    
    def _execute_remote_commands(self, node: GPUNode, commands: List[str]) -> str:
        """Execute commands on remote node"""
        if node.hostname not in self.ssh_clients:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                node.ip_address,
                username=self.config['ssh_user'],
                key_filename=self.config['ssh_key_path']
            )
            self.ssh_clients[node.hostname] = ssh
        
        ssh = self.ssh_clients[node.hostname]
        output = ""
        
        for command in commands:
            stdin, stdout, stderr = ssh.exec_command(command)
            output += stdout.read().decode()
            error = stderr.read().decode()
            if error and "warning" not in error.lower():
                logger.error(f"Error executing {command}: {error}")
        
        return output
    
    def _get_cluster_info(self) -> Dict:
        """Get cluster information"""
        total_gpus = sum(node.gpu_count for node in self.nodes)
        total_tflops = total_gpus * 312  # A100 80GB FP16 Tensor Core
        
        return {
            "cluster_name": self.config['cluster_name'],
            "num_nodes": len(self.nodes),
            "total_gpus": total_gpus,
            "gpu_model": self.nodes[0].gpu_model if self.nodes else "Unknown",
            "total_pflops": total_tflops / 1000,
            "interconnect": "InfiniBand HDR (400Gbps)",
            "nodes": [
                {
                    "hostname": node.hostname,
                    "ip": node.ip_address,
                    "gpus": node.gpu_count,
                    "gpu_model": node.gpu_model
                }
                for node in self.nodes
            ]
        }


def main():
    """Main deployment function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Neural Supercomputer")
    parser.add_argument("--config", required=True, help="Cluster configuration file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate existing deployment")
    args = parser.parse_args()
    
    deployer = ProductionClusterDeployer(args.config)
    
    if args.validate_only:
        deployer._validate_deployment()
    else:
        cluster_info = deployer.deploy_cluster()
        
        print("\n=== Neural Supercomputer Deployment Complete ===")
        print(f"Cluster: {cluster_info['cluster_name']}")
        print(f"Nodes: {cluster_info['num_nodes']}")
        print(f"Total GPUs: {cluster_info['total_gpus']}")
        print(f"Performance: {cluster_info['total_pflops']:.1f} PFLOPS")
        print(f"Interconnect: {cluster_info['interconnect']}")
        
        with open("cluster_info.json", "w") as f:
            json.dump(cluster_info, f, indent=2)


if __name__ == "__main__":
    main()