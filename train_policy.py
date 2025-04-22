from typing import List, Dict, Tuple
from models import GPU, JobInfo

class TrainPolicy:
    def __init__(self):
        pass

    def optimize(self, jobs: List[JobInfo], previous_allocations: Dict[str, List[int]], available_gpus: List[GPU]) -> Dict[str, List[int]]:
        """优化训练任务的分配方案"""
        # 初始化分配方案
        allocation = {}
        #TODO：目前会为已完成的任务分配资源
        # 首先保留已有分配的任务
        for job_info in jobs:
            if job_info.name in previous_allocations:
                # 获取任务当前分配的GPU下标
                current_gpu_indices = previous_allocations[job_info.name]
                allocation[job_info.name] = current_gpu_indices
                # 从可用GPU中移除这些GPU
                for gpu_idx in current_gpu_indices:
                    # 在 available_gpus 中找到 gpu_id 匹配的 GPU
                    gpu_to_remove = next((gpu for gpu in available_gpus if gpu.gpu_id == gpu_idx), None)
                    if gpu_to_remove:
                        available_gpus.remove(gpu_to_remove)
        
        # 为未分配的任务分配GPU
        for job_info in jobs:
            if job_info.name not in allocation:
                # 计算需要的GPU数量
                required_gpus = job_info.num_replicas
                if len(available_gpus) >= required_gpus:
                    # 分配GPU，记录GPU的gpu_id
                    allocated_gpu_indices = [gpu.gpu_id for gpu in available_gpus[:required_gpus]]
                    allocation[job_info.name] = allocated_gpu_indices
                    # 更新可用GPU列表
                    available_gpus = available_gpus[required_gpus:]
                else:
                    # GPU不足，分配所有可用GPU
                    allocated_gpu_indices = [gpu.gpu_id for gpu in available_gpus]
                    allocation[job_info.name] = allocated_gpu_indices
                    available_gpus = []

        return allocation
