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
    def __init__(self, gpu_id: int, node_id: int, is_inference: bool,protect_time,save_time):
        self.gpu_id = gpu_id
        self.node_id = node_id
        self.available_space = 7  # 初始可用空间为7个实例
        self.running_jobs = []  # 当前运行的任务列表

        self.app_cache = {}#字典，记录每个缓存开始保留的时间
        self.is_inference = is_inference  # 是否为推理GPU
        self.state = "FREE"  # 直接使用字符串表示状态
        self.protect_start_time = 0#保护状态开始的时间
        self.borrowed_start_time = 0  # 保护状态开始的时间
        self.save_time=save_time#缓存的保留时间
        self.protect_time=protect_time#GPU保留给推理任务的时间
        self.protect_level=1#保护等级
        self.restart_times=None#训练任务重启次数，暂不考虑

    def can_allocate(self, requested_gpu: int) -> bool:
        return self.available_space >= requested_gpu

    def allocate(self, job, requested_gpu, time):

        self.available_space -= requested_gpu

        self.running_jobs.append(job)
        self.app_cache[job.app] = -1 #添加缓存，-1表示缓存持续存在
        if self.is_inference:
            if job.is_inference:
                self.state = "RUNNING"

                #任务未结束时持续保存缓存
            else:#训练任务使用推理集群的GPU
                self.state="BORROWED"
                self.borrowed_start_time=time

                #print(f"训练任务{job.name}借用GPU{self.gpu_id}")
        else:#训练集群GPU
            self.state='RUNNING'

    def deallocate(self, job: 'Job', requested_gpu, time):

        self.available_space += requested_gpu
        self.running_jobs.remove(job)
        remain_app=set(temp_job.app for temp_job in self.running_jobs)
        if job.app not in remain_app:#没有同类应用继续运行，则缓存保留时间开始倒计时
            self.app_cache[job.app]=time
        if not job.is_inference:#训练任务
            self.state='FREE'#训练任务归还推理集群的GPU也是FREE

            return

        if self.is_inference and not self.running_jobs:
            self.state = "PROTECT"
            self.protect_start_time = time

        #print(f"释放GPU{self.gpu_id}资源完毕，剩余运行任务{[job.name for job in self.running_jobs]}")



class Job_info:
    """用于策略优化的任务信息类"""
    def __init__(self, job, speedup_fn,submit_time, attained_service, num_replicas,max_replicas=None,
                 requested_gpu=7,  duration=-1,run_time=0,preemptible = True):

        self.job=job
        self.is_inference=job.is_inference
        self.speedup_fn=speedup_fn
        self.submit_time=submit_time
        self.application=job.application
        self.num_replicas=num_replicas
        self.min_replicas=0#用于训练任务伸缩，占位
        self.max_replicas=max_replicas#用于训练任务伸缩，占位
        self.requested_gpu=requested_gpu
        self.attained_service=attained_service
        self.preemptible = preemptible#是否可抢占，占位
        self.run_time=run_time
        self.duration=duration
        self.num_restarts=None
        self.age=None
        self.name=job.name

