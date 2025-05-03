from typing import List, Dict, Tuple
from models import GPU
import copy
import logging
import numpy as np

import collections
#输入：推理GPU集群、是否带训练GPU集群暂定、已提交未完成的任务(在优化函数中筛选未分配资源的推理任务)、之前的分配情况
#输出：优化后所有任务的分配方案

#当前策略：先到先服务，优先选择有缓存的GPU
#
LOG = logging.getLogger('infer_scheduler')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# LOG.addHandler(ch)
class CacheFirst:
    def __init__(self):
        self.gpu_state= {}#暂存GPU集群的状态

    def get_gpu_state(self, infer_gpus):#键值对，gpu_id：剩余空间。 后续考虑缓存可能把值变成一个列表
        #对比策略：按何属性排序
        for gpu in infer_gpus:
            if gpu.state=='BORROWED':
                self.gpu_state[gpu.gpu_id] = 0
                continue
            self.gpu_state[gpu.gpu_id]=gpu.available_space


#选择GPU的策略：先找缓存
    def select_gpu(self,job,infer_gpus):#为每个任务选择gpu，TODO：回收逻辑，优先回收缓存相同的训练任务占用的GPU
        gpuID=-1
        for gpu in infer_gpus:#先看看是否存在有缓存且能放下的gpu
            if job.job.app in gpu.app_cache:
                if job.requested_gpu<=self.gpu_state[gpu.gpu_id]:
                    return gpu.gpu_id
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
        LOG.info("InferScheduler optimize")
        LOG.info("prev alloc: %s", prev_alloc)
        self.get_gpu_state(infer_gpus)
        train_jobs=[job for job in job_infos if not job.is_inference]
        infer_jobs = [job for job in job_infos if job.is_inference]

        job_names=[job.name for job in infer_jobs]
        #print("share策略收到任务：",job_names)
        #提取待分配的推理任务
        self.remain_jobs=[jobInfo for jobInfo in infer_jobs if jobInfo.job.status == 'WAIT' or jobInfo.job.status == 'START']

        #按提交时间排序
        self.remain_jobs = sorted(self.remain_jobs, key=lambda x: x.submit_time)
        allocations=copy.deepcopy(prev_alloc)
        for job in self.remain_jobs:

            gpuID=self.select_gpu(job,infer_gpus)
            if gpuID == -1:
                allocations[job.name]=[]#无合适的GPU，需要等待
                print(job.name,"需等待")
            else:
                allocations[job.name]=[gpuID]
                self.gpu_state[gpuID]-=job.requested_gpu

        return allocations