
import copy
import logging
import numpy as np
from collections import OrderedDict
import collections
from typing import List, Dict, Tuple
from models import GPU
from speedup import SpeedupFunction

class DeepBoot(object): # Use DP to calculate
    def __init__(self):
        self._prev_states = None
        self._prev_jobs = None
        self._prev_nodes = None
        self.jobs = None
        self.num_nodes = 16
        self.gpu_each_node=4
        self.total_gpus = None #GPU总数
        self.sched_train = True
        self.infer_schedule = True

    def _get_speedup(self, job, num_replicas):
        # 计算任务在给定副本数下的速度提升
        gpus_each_node = self.gpu_each_node  # 每个节点的 GPU 数量
        num_nodes = num_replicas // gpus_each_node  # 计算需要的节点数
        if num_replicas % gpus_each_node != 0:
            num_nodes += 1  # 如果有余数，增加一个节点

        return job.speedup_fn(num_nodes, num_replicas)  # 返回速度提升

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

    def get_free_gpus(self, total_gpus, allocations):#返回结果是用于训练集群动态规划的GPU列表
        """获取可用的 GPU"""
        # 获取所有已分配的 GPU
        allocated_gpus = set()
        for gpu_indices in allocations.values():
            allocated_gpus.update(gpu_indices)
        
        # 返回未分配的 GPU ID
        return [gpu.gpu_id for gpu in total_gpus if gpu.gpu_id not in allocated_gpus]

    def select_gpus(self,gpu_need,nodes_info):
        '''
                gpu_need: gpus needed by current tasks
                nodes_info: free gpus in each node
        '''

        max_idle_node = max(nodes_info, key=lambda k: len(nodes_info[k]))

        if gpu_need >= len(nodes_info[max_idle_node]):
            return max_idle_node, nodes_info[max_idle_node]

        else:
            f = {k: v for k, v in nodes_info.items() if len(v) >= gpu_need}#能满足任务需求的节点及其GPU列表
            nodes, cnts = list(f.keys()), [len(v) for v in f.values()]
            if not nodes:
                print("异常，找不到满足条件的节点")
                return None, []  # 如果没有满足条件的节点，返回 None 或其他默认值
            node_id = np.argmin(cnts)
            node = nodes[node_id]
            return node, nodes_info[node]

    def replicas2allocation(self, job_names, allocations, num_replicas, available_gpus):
        """将副本数量转换为具体的 GPU 分配方案"""
        # num_replicas为字典

        # 按照副本数排序任务
        job_keys = sorted(job_names, key=lambda k: num_replicas[k])
        # 过滤出副本数匹配的分配，这些任务分配不变
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        # 计算空闲 GPU 数量。
        occupied_gpus = set()#被占用的GPU下标列表
        for gpu_list in allocations.values():
            occupied_gpus.update(gpu_list)
        free_gpus = [gpu for gpu in available_gpus if gpu not in occupied_gpus]#空闲的GPU下标列表

        # 构建每个节点的剩余GPU数量
        nodes_info = {}

        for k in range(self.num_nodes):
            nodes_info[k] = list()
        for gpu in free_gpus:
            nodes_info[gpu // 4].append((gpu))

        for job in job_keys:
            # 为每个任务分配 GPU 资源
            if num_replicas[job] > 0 and not allocations.get(job):#未分配到任务
                allocations[job] = []  # 初始化分配列表
                while len(allocations[job]) < num_replicas[job]:
                    gpu_need = num_replicas[job] - len(allocations[job])#还需要多少GPU
                    node_id,gpu_list = self.select_gpus(gpu_need, nodes_info)#分配到的GPU下标列表
                    num = min(len(gpu_list), gpu_need)#要用多少个GPU
                    allocations[job].extend(gpu_list[:num])
                    nodes_info[node_id] = gpu_list[num:]#更新nodes_infos

        return allocations
        
        #1、根据集群状态获取当前可用的GPU数量，训练集群GPU数量加上推理集群中空闲的GPU数量
        #2、定义每个任务的speed_up计算函数，用于动态规划时的价值计算
        #3、创建每个任务的候选项，即一个任务可能分配多少GPU
        #4、计算speedup作为每种分配方案对应的价值。进行动态规划，得到最优分配方案
        #5、最优分配方案的结构为字典，即任务名：GPU数量。通过replicas2allocation函数得到具体的部署方案train_alloc。
        # train_alloc结构和base_allocations一样。字典，键为job名称,值为gpu_id的列表
        #6、在映射成部署方案的时候，涉及选择节点的逻辑，对应函数为select_node。
        #7、在生成部署方案时，应遵循变动尽可能少的原则，即如果一个任务分配到的GPU数量没有变化，

    def allocate_elastic(self, prev_allocations, jobs, free_gpus): # 弹性资源分配函数
        num_gpus = len(free_gpus)  # 可用 GPU 总数量

        ws = [[]]  # 初始化任务资源矩阵
        vs = [[]]  # 初始化任务价值矩阵

        for job_info in jobs:
            temp_w = []
            temp_v = []
            num_restarts = job_info.num_restarts  # 任务重启次数
            age = job_info.age  # 任务的年龄
            delay = 10  # 延迟参数

            factor = max(age - num_restarts * delay, 0.0) / (age + delay)  # 计算惩罚因子

            for w in range(1, job_info.max_replicas + 1):  # 遍历可能的副本数

                temp_w.append(w)
                speedup = self._get_speedup(job_info, w)  # 计算速度提升

                if job_info.name not in prev_allocations or w != len(prev_allocations[job_info.name]):
                    speedup *= factor  # 应用惩罚因子


                temp_v.append(speedup)

            ws.append(temp_w)
            vs.append(temp_v)

        ways = self.max_value_dp(ws, vs, num_gpus)  # 使用动态规划求解最优解
        num_replicas = {}
        for i, job_info in enumerate(jobs):
            num_replicas[job_info.name] = ways[i]  # # 记录每个任务的副本数
            # 根据副本数重新计算分配
        temp_alloc = copy.deepcopy(prev_allocations)
        job_names = [job.name for job in jobs]
        alloc = self.replicas2allocation(
            job_names,
            allocations=temp_alloc,
            num_replicas=num_replicas,
            available_gpus=free_gpus
        )

        return alloc  # 返回新的分配

    def optimize(self, job_infos, base_allocations,gpus):
        #gpus:整个GPU集群
        """优化训练任务的分配方案"""
        # 1. 获取可用的 GPU,区分训练推理任务

        self.total_gpus=len(gpus)

        def ispinned(job):
            return not job.preemptible and base_allocations.get(job.name, []) != []
        job_infos=sorted(job_infos,
                         key=lambda jobinfo:(not ispinned(jobinfo),
                                             jobinfo.attained_service,
                                             jobinfo.submit_time))

        train_jobs = [job for job in job_infos if not job.is_inference]
        infer_jobs = [job for job in job_infos if job.is_inference]
        infer_job_names=[job.name for job in infer_jobs]

        infer_alloc = {k:v for k,v in base_allocations.items() if "inference" in k}
        prev_train_alloc = {k:v for k,v in base_allocations.items() if "inference" not in k}
        # print("训练优化函数提取infer_alloc",infer_alloc)
        # print("训练优化函数提取train_alloc", prev_train_alloc)


        # 2. 初始化动态规划表,定义job的speedup函数
        #使用pollux的逻辑

        self._job_resources = np.zeros((len(train_jobs), 1), dtype=np.int64)#获取每个任务需要的资源

        for idx, job in enumerate(train_jobs):
            self._job_resources[idx, 0] = 1
        # 构建节点资源矩阵
        len_nodes=self.num_nodes#节点数量
        self._node_resources = np.zeros((len_nodes, 1), dtype=np.int64)
        for k in range(len_nodes):
            self._node_resources[k]=[4]
        gpu_set = set()

        for gpu_list in infer_alloc.values():
            for gpu in gpu_list:
                gpu_set.add(gpu)
        for gpu_id in gpu_set:
            node_id = gpu_id //4
            self._node_resources[node_id]-=1

        shares = self._job_resources / np.sum(self._node_resources, axis=0)#节点GPU总数
        self._dominant_share = np.amax(shares, axis=1)

        # 计算公平分配的副本数和节点数

        fair_replicas = np.ceil(1.0 / self._dominant_share / len(train_jobs))
        fair_nodes = np.ceil(len_nodes * self._dominant_share)

        # 更新任务的速度提升函数
        for job, num_nodes, num_replicas in zip(train_jobs, fair_nodes, fair_replicas):
            if not hasattr(job.speedup_fn, "_goodput_fn"):

                job.speedup_fn = lambda n, r: r / num_replicas
                continue

            job.speedup_fn._base_goodput = job.speedup_fn._goodput_fn.optimize(
                num_nodes=num_nodes, num_replicas=max(num_replicas, num_nodes),
                max_batch_size=job.speedup_fn._max_batch_size,
                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                accumulation=job.speedup_fn._accumulation)[0]
        # 套用DeepBoot的逻辑
        allocations = {}

        # 3. 获取最优分配方案
        free_gpus = self.get_free_gpus(gpus, infer_alloc)  # 减去推理集群被占用的资源
        #print("free gpu",free_gpus)
        train_alloc = self.allocate_elastic(prev_train_alloc,train_jobs,free_gpus)
        allocations.update(train_alloc)
        return allocations

