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
    def __init__(self, gpu_id: int, node_id: int, is_inference: bool,save_time=30):
        self.gpu_id = gpu_id
        self.node_id = node_id
        self.available_space = 100  # 初始可用空间为100%
        self.running_jobs = []  # 当前运行的任务列表
        self.application_cache=set()
        self.is_inference = is_inference  # 是否为推理GPU
        self.state = "FREE"  # 直接使用字符串表示状态
        self.protect_start_time = 0
        self.save_time=save_time#任务运行完后的保留时间
        self.protect_level=1#保护等级
        self.restart_times=None#训练任务重启次数，暂不考虑

    def can_allocate(self, requested_gpu: int) -> bool:
        return self.available_space >= requested_gpu

    def allocate(self, job, requested_gpu: int):

        self.available_space -= requested_gpu
        self.running_jobs.append(job)

        if self.is_inference:
            if job.is_inference:
                self.state = "RUNNING"
                self.application_cache.add(job.app)  # 添加缓存
            else:#训练任务使用推理集群的GPU
                self.state="BORROWED"
        else:#训练集群GPU
            self.state='RUNNING'

    def deallocate(self, job: 'Job', requested_gpu: int):
        print("原运行任务列表：",[job.name for job in self.running_jobs])
        self.available_space += requested_gpu
        self.running_jobs.remove(job)
        print(f"任务{job.name}释放GPU{self.gpu_id}资源")
        if not job.is_inference:#训练任务

            self.state='FREE'#训练任务归还推理集群的GPU也是FREE

        if self.is_inference and not self.running_jobs:
            self.state = "PROTECT"
            self.protect_start_time = 0  # 这个值会在Cluster类中设置
        print(f"释放GPU{self.gpu_id}资源完毕，{self.available_space}，{self.running_jobs}")

class JobInfo:
    """用于策略优化的任务信息类"""
    def __init__(self, name, submit_time, application, num_replicas,
                 requested_gpu: int, batch_size: int, duration: int,
                 is_inference: bool, node_gpu_distribution: List[int], placement: List[int],end_time):
        self.name = name
        self.submit_time = submit_time
        self.application = application
        self.num_replicas = num_replicas
        self.requested_gpu = requested_gpu
        self.batch_size = batch_size
        self.duration = duration
        self.is_inference = is_inference
        self.node_gpu_distribution = node_gpu_distribution
        self.placement = placement
        self.end_time=end_time

class Job_info:
    """用于策略优化的任务信息类"""
    def __init__(self, job, speedup_fn,submit_time, num_replicas,
                 requested_gpu: int,  duration=-1,run_time=0):

        self.job=job
        self.is_inference=job.is_inference
        self.speedup_fn=speedup_fn
        self.submit_time=submit_time
        self.application=job.application
        self.num_replicas=num_replicas
        self.min_replicas=0#用于训练任务伸缩，占位
        self.max_replicas=num_replicas#用于训练任务伸缩，占位
        self.requested_gpu=requested_gpu
        self.preemptible = False#是否可抢占，占位
        self.run_time=run_time
        self.duration=duration

        self.name=job.name
