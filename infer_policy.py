from typing import List, Dict, Tuple
from models import GPU, JobInfo

class InferPolicy:
    def __init__(self):
        pass

    def optimize(self, jobs: List[JobInfo], prev_alloc,gpus) :
        """
        批量优化推理任务分配
        Args:
            jobs: 所有任务列表
            cluster: 集群对象
        Returns:
            Tuple[Dict[str, List[GPU]], Dict[GPU, List[str]]]: 
                - 任务名到GPU列表的映射
                - GPU到任务名列表的映射
        """
        # 筛选出未完成的推理任务
        infer_jobs = [job for job in jobs if job.end_time is None]
        
        # 获取所有可用的推理GPU
        available_gpus = [gpu for gpu in gpus if gpu.state == "FREE"]

        # 初始化分配结果
        job_to_gpus = {}  # 任务到GPU的映射
        gpu_to_jobs = {gpu: [] for gpu in available_gpus}  # GPU到任务的映射
        
        for job in infer_jobs:
            # 每个推理任务只需要1个GPU
            needed_gpus = 1
            
            # 尝试在现有GPU上分配
            allocated_gpus = []
            for gpu in available_gpus:
                # 检查GPU是否有足够的剩余空间
                if gpu.available_space >= job.requested_gpu:
                    allocated_gpus.append(gpu.gpu_id)
                    gpu_to_jobs[gpu].append(job.name)
                    if len(allocated_gpus) == needed_gpus:
                        break
            
            # 如果成功分配了足够的GPU
            if len(allocated_gpus) == needed_gpus:
                job_to_gpus[job.name] = allocated_gpus
        
        return job_to_gpus, gpu_to_jobs
