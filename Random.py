from typing import List, Dict, Tuple
from models import GPU
import copy
import logging
import numpy as np
import random
import collections
#输入：推理GPU集群、是否带训练GPU集群暂定、已提交未完成的任务(在优化函数中筛选未分配资源的推理任务)、之前的分配情况
#输出：优化后所有任务的分配方案

#当前策略：先到先服务，随机分配
#
LOG = logging.getLogger('infer_scheduler')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# LOG.addHandler(ch)
class Random:
    def __init__(self, max_wait_time=10):
        self.gpu_state = {}  # 暂存GPU集群的状态
        self.borrowed_gpus = []
        self.max_wait_time = max_wait_time  # 容忍排队的最长时间

    def get_gpu_state(self, infer_gpus):#键值对，gpu_id：剩余空间。 后续考虑缓存可能把值变成一个列表
        #对比策略：按何属性排序
        for gpu in infer_gpus:
            if gpu.state=='BORROWED':
                self.gpu_state[gpu.gpu_id] = 0
                continue
            self.gpu_state[gpu.gpu_id]=gpu.available_space


#选择GPU的策略：随机分配
    def select_gpu(self,job):#为每个任务选择gpu
        gpu_list=[]#满足任务需求的ID列表
        for gpu_id,space in self.gpu_state.items():
            if space>=job.requested_gpu:
                gpu_list.append(gpu_id)
        gpuID=-1
        if gpu_list:
            gpuID = random.choice(gpu_list)#随机选择一个GPU
        return gpuID


    def optimize(self, job_infos, prev_alloc, infer_gpus,preemptible=False) :
        """
        批量优化推理任务分配
        Args:
            jobs: 已提交未完成的任务
            prev_alloc:已分配过alloc但未完成的任务的分配，不含新任务

        """
        LOG.info("InferScheduler optimize")
        LOG.info("prev alloc: %s", prev_alloc)
        self.get_gpu_state(infer_gpus)
        self.borrowed_gpus = [gpu for gpu in infer_gpus if gpu.state == 'BORROWED']  # 借出去的GPU
        # 此处可添加排序策略开关
        self.borrowed_gpus = sorted(self.borrowed_gpus, key=lambda gpu: (gpu.borrowed_start_time))  # 按借出时间早晚排序
        train_jobs=[job for job in job_infos if not job.is_inference]
        infer_jobs = [job for job in job_infos if job.is_inference]

        job_names=[job.name for job in infer_jobs]
        #print("share策略收到任务：",job_names)
        #提取待分配的推理任务
        self.remain_jobs=[jobInfo for jobInfo in infer_jobs if jobInfo.job.status == 'WAIT' or jobInfo.job.status == 'START']

        #按提交时间排序
        self.remain_jobs = sorted(self.remain_jobs, key=lambda x: x.submit_time)

        allocations=copy.deepcopy(prev_alloc)
        reclaim_event=0
        for job in self.remain_jobs:
            gpuID=self.select_gpu(job)
            if gpuID == -1:
                if preemptible:
                    # print(f"{job.name}未找到合适GPU,已排队{job.age}")
                    if job.age <= job.job.max_wait or len(self.borrowed_gpus) == 0:  # 未超过容忍的排队时间
                        allocations[job.name] = []  # 无合适的GPU，需要等待
                    else:  # 回收一个GPU
                        if len(self.borrowed_gpus)==0:
                            allocations[job.name] = []
                            continue
                        reclaim_gpu = self.borrowed_gpus[0]

                        train_job = reclaim_gpu.running_jobs[0]  # 理论上只会借给一个训练任务
                        self.gpu_state[reclaim_gpu.gpu_id] = 7
                        self.gpu_state[reclaim_gpu.gpu_id] -= job.requested_gpu
                        allocations[job.name] = [reclaim_gpu.gpu_id]
                        allocations[train_job.name].remove(reclaim_gpu.gpu_id)  # 修改对应训练任务的alloc
                        LOG.info(f"{job.name}排队时间过长，回收借给训练任务{train_job.name}的GPU{reclaim_gpu.gpu_id}")
                        print(f"{job.name}排队时间过长，回收借给训练任务{train_job.name}的GPU{reclaim_gpu.gpu_id}")
                        self.borrowed_gpus.remove(reclaim_gpu)  # 回收的GPU移出BORROWED列表
                        reclaim_event+=1
                else:
                    allocations[job.name] = []  # 无合适的GPU，需要等待

            else:
                allocations[job.name]=[gpuID]
                self.gpu_state[gpuID]-=job.requested_gpu

        return allocations, reclaim_event