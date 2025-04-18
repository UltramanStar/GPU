from enum import Enum
from typing import List, Dict, Optional
from old_application import APPLICATIONS, APPLICATIONS_DELAY, FIRST_DELAY, NEXT_DELAY

def app_trans(app_name):
    app_name_split = app_name.split('-')
    # LOG.info("app name split: %s")
    if len(app_name_split) == 2 or 'infer' not in app_name:
        return app_name
    else:
        return "-".join(app_name_split[:-1])

class GPU:
    def __init__(self, gpu_id: int, node_id: int, is_inference: bool):
        self.gpu_id = gpu_id
        self.node_id = node_id
        self.available_space = 100  # 初始可用空间为100%
        self.running_jobs = []  # 当前运行的任务列表
        self.is_inference = is_inference  # 是否为推理GPU
        self.state = "FREE"  # 直接使用字符串表示状态
        self.protect_start_time = 0
        self.protect_level=1#保护等级
        self.restart_times=None#训练任务重启次数，暂不考虑

    def can_allocate(self, requested_gpu: int) -> bool:
        return self.available_space >= requested_gpu

    def allocate(self, job: 'Job', requested_gpu: int):
        self.available_space -= requested_gpu
        self.running_jobs.append(job)
        if self.is_inference:
            self.state = "RUNNING"

    def deallocate(self, job: 'Job', requested_gpu: int):
        self.available_space += requested_gpu
        self.running_jobs.remove(job)
        if not job.is_inference:
            print("训练任务释放资源，剩余任务：",self.running_jobs)
        if self.is_inference and not self.running_jobs:
            self.state = "PROTECT"
            self.protect_start_time = 0  # 这个值会在Cluster类中设置

class JobInfo:
    """用于策略优化的任务信息类"""
    def __init__(self, name: str, submit_time: int, application, num_replicas: int,
                 requested_gpu: int, batch_size: int, duration: int, num_task: int,
                 is_inference: bool, node_gpu_distribution: List[int], placement: List[int],end_time):
        self.name = name
        self.submit_time = submit_time
        self.application = application
        self.num_replicas = num_replicas
        self.requested_gpu = requested_gpu
        self.batch_size = batch_size
        self.duration = duration
        self.num_task = num_task
        self.is_inference = is_inference
        self.node_gpu_distribution = node_gpu_distribution
        self.placement = placement
        self.end_time=end_time