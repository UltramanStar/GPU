from typing import List, Dict, Tuple
from models import GPU, JobInfo
import copy
import logging
import numpy as np

import collections
#输入：推理GPU集群、是否带训练GPU集群暂定、已提交未完成的任务(在优化函数中筛选未分配资源的推理任务)、之前的分配情况
#输出：优化后所有任务的分配方案

#当前策略：先到先服务，FirstFit
#

class Share:
    def __init__(self):
        self.gpu_state= {}#暂存GPU集群的状态

    def get_gpu_state(self, infer_gpus):#键值对，gpu_id：剩余空间。 后续考虑缓存可能把值变成一个列表
        for gpu in infer_gpus:
            if gpu.state=='BORROWED':
                self.gpu_state[gpu.gpu_id] = 0
                continue
            self.gpu_state[gpu.gpu_id]=gpu.available_space



    def select_gpu(self,job):#为每个任务选择gpu
        gpuID=-1
        for gpu_id,space in self.gpu_state.items():
            if space>=job.requested_gpu:
                gpuID=gpu_id
                break
        return gpuID



    def optimize(self, job_infos, prev_alloc, infer_gpus) :
        """
        批量优化推理任务分配
        Args:
            jobs: 已提交未完成的任务
            prev_alloc:已分配过alloc但未完成的任务的分配，不含新任务

        """

        self.get_gpu_state(infer_gpus)
        train_jobs=[job for job in job_infos if not job.is_inference]
        infer_jobs = [job for job in job_infos if job.is_inference]
        #提取待分配的推理任务
        self.remain_jobs=[jobInfo for jobInfo in infer_jobs if jobInfo.job.status == 'WAIT' or jobInfo.job.status == 'START']

        #按提交时间排序
        self.remain_jobs = sorted(self.remain_jobs, key=lambda x: x.submit_time)
        allocations=copy.deepcopy(prev_alloc)
        for job in self.remain_jobs:
            gpuID=self.select_gpu(job)
            if gpuID == -1:
                allocations[job.name]=[]#无合适的GPU，需要等待
            else:
                allocations[job.name]=[gpuID]
                self.gpu_state[gpuID]-=job.requested_gpu

        return allocations