from typing import List, Dict, Tuple
from models import GPU
import copy
import logging
import numpy as np
import random
import collections
#输入：推理GPU集群、是否带训练GPU集群暂定、已提交未完成的任务(在优化函数中筛选未分配资源的推理任务)、之前的分配情况
#输出：优化后所有任务的分配方案

#当前策略：先到先服务，批量处理，减少碎片化
#
LOG = logging.getLogger('infer_scheduler')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# LOG.addHandler(ch)
class BestFit:
    def __init__(self, max_wait_time=10):
        self.gpu_state = {}  # 暂存GPU集群的状态
        self.gpu_cache = {}  # 暂存GPU集群的缓存状态
        self.borrowed_gpus = []
        self.max_wait_time = max_wait_time  # 容忍排队的最长时间

    def get_gpu_state(self, infer_gpus):#键值对，gpu_id：剩余空间。 后续考虑缓存可能把值变成一个列表
        #对比策略：按何属性排序
        for gpu in infer_gpus:
            self.gpu_cache[gpu.gpu_id]=set(gpu.app_cache.keys())#记录有的缓存类型
            if gpu.state=='BORROWED':
                self.gpu_state[gpu.gpu_id] = 0
                continue
            self.gpu_state[gpu.gpu_id]=gpu.available_space


#选择GPU的策略：最佳匹配
    def select_gpu(self,job):#为每个任务选择gpu
        gpu_list=[]#满足任务需求的ID列表
        for gpu_id,space in self.gpu_state.items():
            if space>=job.requested_gpu:
                gpu_list.append(gpu_id)
        if len(gpu_list) == 0:
            return -1
        gpu_info = []
        for gpu_id in gpu_list:
            # 剩余空间最少的优先（因此用负数表示，以便升序排序时等同于降序）
            remaining_space = self.gpu_state[gpu_id]
            # 检查是否有缓存（True/False）
            has_cache = job.job.app in self.gpu_cache.get(gpu_id, set())
            # 添加到列表中，用于排序
            gpu_info.append((remaining_space, not has_cache, gpu_id))


        # 然后按是否有缓存（没有缓存的排在后面，因此用not has_cache，False排在True前面）
        # 最后按GPU下标升序
        gpu_info.sort()
        # 返回排序后第一个GPU的ID
        return gpu_info[0][2]

    def get_num_reclaim(self,remaining_jobs):#计算需要回收的GPU数量
        """
        计算需要多少新的空GPU（每个提供7份）才能完成剩余任务分配。
        核心逻辑：模拟贪心装箱过程，优先填充大任务减少碎片。
        """
        # 将任务按所需GPU份数降序排列（7,4,3,2,1）
        sorted_tasks = sorted(remaining_jobs,
                              key=lambda x: -x.requested_gpu)
        new_gpus = []  # 记录每个新GPU的剩余容量

        for task in sorted_tasks:
            required = task.requested_gpu
            allocated = False
            # 尝试将任务放入已有新GPU
            for i in range(len(new_gpus)):
                if new_gpus[i] >= required:
                    new_gpus[i] -= required
                    allocated = True
                    break
            # 无法放入已有GPU时，创建新GPU
            if not allocated:
                new_gpus.append(7 - required)  # 新GPU的剩余空间

        return len(new_gpus)

    def optimize(self, job_infos, prev_alloc, infer_gpus,preemptible=False) :
        """
        批量优化推理任务分配
        Args:
            jobs: 已提交未完成的任务
            prev_alloc:已分配过alloc但未完成的任务的分配，不含新任务

        """
        LOG.info("InferScheduler optimize")
        LOG.info("prev alloc: %s", prev_alloc)
        self.get_gpu_state(infer_gpus)#记录剩余空间和缓存情况
        self.borrowed_gpus = [gpu for gpu in infer_gpus if gpu.state == 'BORROWED']  # 借出去的GPU
        # 此处可添加排序策略开关
        self.borrowed_gpus = sorted(self.borrowed_gpus, key=lambda gpu: (gpu.borrowed_start_time))  # 按借出时间早晚排序
        train_jobs=[job for job in job_infos if not job.is_inference]
        infer_jobs = [job for job in job_infos if job.is_inference]


        #提取待分配的推理任务
        self.remain_jobs=[jobInfo for jobInfo in infer_jobs if jobInfo.job.status == 'WAIT' or jobInfo.job.status == 'START']

        #先按大小降序排序，再按提交时间排序
        self.remain_jobs = sorted(self.remain_jobs,
                                  key=lambda x: (-x.requested_gpu,x.submit_time))#负号用于降序排列
        allocations=copy.deepcopy(prev_alloc)

        reclaim_event=0
        #1、先处理7份GPU的任务中找得到缓存的，尽量分配到有缓存的
        remove_jobs=[]
        for job in self.remain_jobs:
            if job.requested_gpu!=7:
                break
            available_gpus = [k for k,v in self.gpu_state.items() if v == 7]#剩余份数为7的gpuID
            for gpu in available_gpus:
                if job.job.app in self.gpu_cache[gpu]:
                    allocations[job.name] = [gpu]
                    self.gpu_state[gpu] -= job.requested_gpu
                    remove_jobs.append(job)
                    break
        for job in remove_jobs:#把分配完的任务删去

            self.remain_jobs.remove(job)
        #2、从大到小处理任务，先筛选满足条件的GPU，从中选择剩余空间最小的，有多个则先选有缓存的
        wait_jobs=[]
        for job in self.remain_jobs:

            gpuID=self.select_gpu(job)
            if gpuID == -1:
                allocations[job.name] = []  # 无合适的GPU，需要等待
                if preemptible:
                    wait_jobs.append(job)
                else:
                    print(job.name, "需等待")
            else:
                allocations[job.name] = [gpuID]
                self.gpu_state[gpuID] -= job.requested_gpu
                self.gpu_cache[gpuID].add(job.job.app)#添加缓存，便于相同应用的任务分配到一起
        #3、如果没有满足条件的GPU，说明资源分配完了，选择排队时间较长的任务，计算强制回收的GPU数量，再重复上述步骤
        if wait_jobs and len(self.borrowed_gpus)>0:#允许抢占并且有任务没分配到资源,回收资源
            long_wait=[job for job in wait_jobs if job.age >= job.job.max_wait]#长时间排队的推理任务
            if long_wait:
                print("长时间等待的任务数量：",len(long_wait))
                additinoal_need=self.get_num_reclaim(long_wait)
                reclaim=self.borrowed_gpus[:min(additinoal_need,len(self.borrowed_gpus))]#回收的GPU列表
                reclaim_event+=len(reclaim)
                for gpu in reclaim:
                    train_job = gpu.running_jobs[0]  # 理论上只会借给一个训练任务
                    self.gpu_state[gpu.gpu_id] = 7
                    allocations[train_job.name].remove(gpu.gpu_id)
                    self.borrowed_gpus.remove(gpu)
                    LOG.info(f"回收借给训练任务{train_job.name}的GPU{gpu.gpu_id}")
                    print(f"回收借给训练任务{train_job.name}的GPU{gpu.gpu_id}")

                for job in long_wait:
                    gpuID = self.select_gpu(job)
                    if gpuID == -1:
                        allocations[job.name] = []  # 无合适的GPU，需要等待
                        print(job.name, "需等待")
                    else:
                        allocations[job.name] = [gpuID]
                        self.gpu_state[gpuID] -= job.requested_gpu
        return allocations,reclaim_event

