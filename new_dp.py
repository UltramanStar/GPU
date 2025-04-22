
import copy
import logging
import numpy as np
from collections import OrderedDict
import collections
from typing import List, Dict, Tuple
from models import GPU, JobInfo
from speedup import SpeedupFunction

class DeepBoot(object): # Use DP to calculate
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        self.jobs = None
        self.nodes = None
        self.total_gpus = None
        self.sched_train = True
        self.infer_schedule = True

    def _get_speedup(self, job, num_replicas):
        # LOG.info("total gpus: %s",self.total_gpus)
        gpus_each_node = list(self.total_gpus.values())[0]
        num_nodes = num_replicas // gpus_each_node
        if num_replicas % gpus_each_node != 0:
            num_nodes += 1

        return job.speedup_fn(num_nodes, num_replicas)

    def max_value_dp(self, ws, vs, m):
        # group knapsack
        n = len(ws) - 1
        dp = np.zeros(shape=(n + 1, m + 1))
        for i in range(1, n + 1):  # 任务
            for j in range(m, -1, -1):
                dp[i][j] = dp[i - 1][j]
                for k in range(len(ws[i])):
                    if j >= ws[i][k]:
                        dp[i][j] = max(dp[i][j], dp[i - 1][j - ws[i][k]] + vs[i][k])

        j = m
        ways = np.zeros(n + 1, dtype=int)
        for i in range(n, 0, -1):
            for k in range(len(ws[i])):
                if j >= ws[i][k] and dp[i][j] == dp[i - 1][j - ws[i][k]] + vs[i][k]:
                    ways[i] = ws[i][k]  # limitation of task k
                    j -= ws[i][k]
                    break

        return ways[1:]

    def get_free_gpus(self, total_gpus, allocations):
        """获取可用的 GPU"""
        # 获取所有已分配的 GPU
        allocated_gpus = set()
        for gpu_indices in allocations.values():
            allocated_gpus.update(gpu_indices)
        
        # 返回未分配的 GPU
        return [gpu for gpu in total_gpus if gpu.gpu_id not in allocated_gpus]

    def replicas2allocation(self, jobs, allocations, num_replicas, available_gpus):
        """将副本数量转换为具体的 GPU 分配方案"""
        # 初始化分配方案
        new_allocations = {}
        
        # 首先处理已有分配的任务
        for job in jobs:
            if job.name in allocations:
                # 如果任务已有分配且副本数量未变，保持原有分配
                if num_replicas.get(job.name, 0) == len(allocations[job.name]):
                    new_allocations[job.name] = allocations[job.name]
                    # 从可用 GPU 中移除已分配的 GPU
                    for gpu_id in allocations[job.name]:
                        available_gpus = [gpu for gpu in available_gpus if gpu.gpu_id != gpu_id]
        
        # 处理需要新分配的任务
        for job in jobs:
            if job.name not in new_allocations and job.name in num_replicas:
                required_gpus = num_replicas[job.name]
                if len(available_gpus) >= required_gpus:
                    # 分配 GPU
                    allocated_gpus = available_gpus[:required_gpus]
                    new_allocations[job.name] = [gpu.gpu_id for gpu in allocated_gpus]
                    available_gpus = available_gpus[required_gpus:]
        
        return new_allocations
        
        #1、根据集群状态获取当前可用的GPU数量，训练集群GPU数量加上推理集群中空闲的GPU数量
        #2、定义每个任务的speed_up计算函数，用于动态规划时的价值计算
        #3、创建每个任务的候选项，即一个任务可能分配多少GPU
        #4、计算speedup作为每种分配方案对应的价值。进行动态规划，得到最优分配方案
        #5、最优分配方案的结构为字典，即任务名：GPU数量。通过replicas2allocation函数得到具体的部署方案train_alloc。
        # train_alloc结构和base_allocations一样。字典，键为job名称,值为gpu_id的列表
        #6、在映射成部署方案的时候，涉及选择节点的逻辑，对应函数为select_node。
        #7、在生成部署方案时，应遵循变动尽可能少的原则，即如果一个任务分配到的GPU数量没有变化，
    def optimize(self, jobs, gpus, base_allocations, clock=None):
        """优化训练任务的分配方案"""
        # 1. 获取可用的 GPU
        available_gpus = self.get_free_gpus(gpus, base_allocations)
        num_available_gpus = len(available_gpus)
        
        # 2. 初始化动态规划表
        dp = [[0.0 for _ in range(num_available_gpus + 1)] for _ in range(len(jobs) + 1)]
        path = [[{} for _ in range(num_available_gpus + 1)] for _ in range(len(jobs) + 1)]
        
        # 3. 动态规划填充表格
        for i in range(1, len(jobs) + 1):
            job = jobs[i - 1]
            for j in range(1, num_available_gpus + 1):
                for k in range(1, min(job.num_replicas + 1, j + 1)):
                    current_speedup = self.speedup_fn(job, k)
                    if dp[i - 1][j - k] + current_speedup > dp[i][j]:
                        dp[i][j] = dp[i - 1][j - k] + current_speedup
                        path[i][j] = path[i - 1][j - k].copy()
                        path[i][j][job.name] = k
        
        # 4. 获取最优分配方案
        optimal_replicas = path[len(jobs)][num_available_gpus]
        
        # 5. 转换为具体的 GPU 分配方案
        train_alloc = self.replicas2allocation(jobs, base_allocations, optimal_replicas, available_gpus)
        
        return train_alloc

