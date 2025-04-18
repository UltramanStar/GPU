from typing import List, Dict, Tuple
from models import GPU, JobInfo

class TrainPolicy:
    def __init__(self):
        pass

    def optimize(self, jobs: List[JobInfo], previous_allocation: Dict[str, List[GPU]], available_gpus: List[GPU]) -> Dict[str, List[GPU]]:
        """优化训练任务的分配方案"""
        # 初始化分配方案
        allocation = {}
        
        # 首先保留已有分配的任务
        for job_info in jobs:
            if job_info.name in previous_allocation:
                # 获取任务当前分配的GPU
                current_gpus = previous_allocation[job_info.name]
                allocation[job_info.name] = current_gpus
                # 从可用GPU中移除这些GPU
                for gpu in current_gpus:
                    if gpu in available_gpus:
                        available_gpus.remove(gpu)
        
        # 为未分配的任务分配GPU
        for job_info in jobs:
            if job_info.name not in allocation:
                # 计算需要的GPU数量
                required_gpus = job_info.num_replicas
                if len(available_gpus) >= required_gpus:
                    # 分配GPU
                    allocated_gpus = available_gpus[:required_gpus]
                    allocation[job_info.name] = allocated_gpus
                    # 更新可用GPU列表
                    available_gpus = available_gpus[required_gpus:]
                else:
                    # GPU不足，分配所有可用GPU
                    allocation[job_info.name] = available_gpus
                    available_gpus = []
        
        return allocation
