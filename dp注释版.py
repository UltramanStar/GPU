import copy
import logging
import numpy as np
from collections import OrderedDict
import collections

LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class DeepBoot(object):  # Use DP to calculate
    def __init__(self):
        self._prev_states = None  # 用于存储前一个状态
        self._prev_jobs = None  # 用于存储前一个任务列表
        self._prev_nodes = None  # 用于存储前一个节点列表
        self.jobs = None  # 当前任务列表
        self.nodes = None  # 当前节点列表
        self.total_gpus = None  # 集群总 GPU 数量
        self.sched_train = True  # 是否调度训练任务
        self.infer_schedule = True  # 是否调度推理任务

    def select_node(self, num_replica, free_gpus):
        # num_replica: 当前任务需要的 GPU 数量
        # free_gpus: 每个节点上可用的 GPU 数量
        ORIGIN_SELECT = False  # False 表示使用新的节点选择方案

        # 使用 Counter 统计每个节点的空闲 GPU 数量
        node_idx, count = free_gpus.most_common(1)[0]
        if ORIGIN_SELECT:
            return node_idx, count  # 返回节点索引和空闲 GPU 数量

        # 如果请求的 GPU 数量超过当前最大空闲 GPU 节点的数量，则返回该节点及其 GPU 数量
        if num_replica > count:
            return node_idx, count
        else:
            # 过滤出能够满足当前任务 GPU 需求的节点
            f = {k:v for k,v in dict(free_gpus).items() if v >= num_replica}
            nodes, cnts = list(f.keys()), list(f.values())  # 获取可用节点及其 GPU 数量
            node_id = np.argmin(cnts)  # 找到 GPU 数量最少的节点
            node = nodes[node_id]
            return node, cnts[node_id]  # 返回节点及其 GPU 数量

    def replicas2allocation(self, jobs, allocations, num_replicas, available_gpus):
        # jobs: 当前任务列表
        # allocations: 当前的资源分配
        # num_replicas: 每个任务需要的副本数
        # available_gpus: 可用的 GPU 资源

        # 按照副本数排序任务
        job_keys = sorted(jobs, key=lambda k: num_replicas[k])
        # 过滤出副本数匹配的分配
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        # 计算每个节点的空闲 GPU 数量
        free_gpus = collections.Counter(available_gpus) - collections.Counter(sum(allocations.values(), []))

        for key in job_keys:
            # 为每个任务分配 GPU 资源
            if num_replicas[key] > 0 and not allocations.get(key):
                allocations[key] = []  # 初始化分配列表
                while len(allocations[key]) < num_replicas[key]:
                    gpu_need = num_replicas[key] - len(allocations[key])  # 计算还需要的 GPU 数量
                    node_idx, count = self.select_node(gpu_need, free_gpus)  # 选择节点
                    num = min(count, gpu_need)  # 计算实际可以分配的 GPU 数量
                    allocations[key].extend([node_idx] * num)  # 更新分配
                    free_gpus[node_idx] -= num  # 更新空闲 GPU 数量

        return allocations  # 返回更新后的分配

    def _get_speedup(self, job, num_replicas):
        # 计算任务在给定副本数下的速度提升
        gpus_each_node = list(self.total_gpus.values())[0]  # 每个节点的 GPU 数量
        num_nodes = num_replicas // gpus_each_node  # 计算需要的节点数
        if num_replicas % gpus_each_node != 0:
            num_nodes += 1  # 如果有余数，增加一个节点

        return job.speedup_fn(num_nodes, num_replicas)  # 返回速度提升

    def max_value_dp(self, ws, vs, m):
        # 动态规划算法，用于求解背包问题
        n = len(ws) - 1  # 任务数量
        dp = np.zeros(shape=(n + 1, m + 1))  # 初始化动态规划表

        for i in range(1, n + 1):  # 遍历每个任务
            for j in range(m, -1, -1):  # 遍历资源限制，从大到小
                dp[i][j] = dp[i - 1][j]  # 初始化为不选择当前任务的情况
                for k in range(len(ws[i])):  # 遍历当前任务的所有可能资源分配
                    if j >= ws[i][k]:
                        # 更新最大值
                        dp[i][j] = max(dp[i][j], dp[i - 1][j - ws[i][k]] + vs[i][k])

        # 回溯路径，找到最优解
        j = m
        ways = np.zeros(n + 1, dtype=int)  # 用于存储每个任务的分配
        for i in range(n, 0, -1):
            for k in range(len(ws[i])):
                if j >= ws[i][k] and dp[i][j] == dp[i - 1][j - ws[i][k]] + vs[i][k]:
                    ways[i] = ws[i][k]  # 记录当前任务的资源分配
                    j -= ws[i][k]  # 更新剩余资源
                    break

        return ways[1:]  # 返回最优解

    def allocate_elastic(self, prev_allocations, jobs, free_gpus):
        # 弹性资源分配函数
        num_gpus = sum(free_gpus.values())  # 可用 GPU 总数量
        ws = [[]]  # 初始化任务资源矩阵
        vs = [[]]  # 初始化任务价值矩阵

        for job, info in jobs.items():
            temp_w = []
            temp_v = []
            num_restarts = info.num_restarts  # 任务重启次数
            age = info.age  # 任务的年龄
            delay = 10  # 延迟参数
            factor = max(age - num_restarts * delay, 0.0) / (age + delay)  # 计算惩罚因子

            for w in range(1, info.max_replicas + 1):  # 遍历可能的副本数
                temp_w.append(w)
                speedup = self._get_speedup(info, w)  # 计算速度提升
                if job not in prev_allocations or w != len(prev_allocations[job]):
                    speedup *= factor  # 应用惩罚因子
                temp_v.append(speedup)

            ws.append(temp_w)
            vs.append(temp_v)

        ways = self.max_value_dp(ws, vs, num_gpus)  # 使用动态规划求解最优解
        num_replicas = {}
        for i, job in enumerate(jobs):
            num_replicas[job] = ways[i]  # 记录每个任务的副本数

        # 根据副本数重新计算分配
        temp_alloc = copy.deepcopy(prev_allocations)
        alloc = self.replicas2allocation(
            jobs=jobs,
            allocations=temp_alloc,
            num_replicas=num_replicas,
            available_gpus=free_gpus
        )

        return alloc  # 返回新的分配

    def infer_pod_status_trans(self, infer_pod_status):
        # 转换推理任务状态格式
        _infer_pod_status = dict()
        for _, pods in infer_pod_status.items():
            for name, info in pods.items():
                _infer_pod_status[name] = info
        self.infer_pod_status = _infer_pod_status

    def get_free_gpus(self, total_gpus, allocations):
        # 计算空闲 GPU 数量
        return collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))

    def optimize(self, jobs, nodes, base_allocations, node_template, clock=None, infer_pod_status=None):
        # 主调度函数，优化资源分配
        def ispinned(key, job):
            return not job.preemptible and base_allocations.get(key, []) != []

        sleep_pods = set()  # 睡眠任务集合
        infer_pods = set()  # 推理任务集合
        self.total_gpus = total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}  # 集群总 GPU 资源
        self.infer_pod_status_trans(infer_pod_status)  # 转换推理任务状态
        self.node_id_dict = dict(zip(nodes.keys(), range(len(nodes))))  # 节点索引字典

        prev_allocations = base_allocations  # 前一个分配状态

        # 初始化任务列表
        self.jobs = jobs = OrderedDict(sorted(jobs.items(),
                                          key=lambda kv: (not ispinned(kv[0], kv[1]),
                                                          kv[1].attained_service,
                                                          kv[1].creation_timestamp)))

        # 初始化节点列表，按是否可抢占排序
        self.nodes = nodes = OrderedDict(sorted(nodes.items(), key=lambda kv: (kv[1].preemptible, kv[0])))

        # 分离训练任务和推理任务
        train_jobs = {}
        infer_jobs = {}
        sleep_jobs = {}
        if self.infer_schedule:
            train_jobs = {k: v for k, v in jobs.items() if not v.inference}  # 训练任务
            infer_jobs = {k: v for k, v in jobs.items() if v.inference and k not in sleep_pods}  # 推理任务
            sleep_jobs = {k: v for k, v in jobs.items() if k in sleep_pods and k in prev_allocations}  # 睡眠任务
        else:
            train_jobs = jobs

        if len(train_jobs) == 0:
            return prev_allocations, len(nodes)  # 如果没有训练任务，返回前一个分配

        self._jobs = train_jobs  # 当前训练任务
        LOG.info("prev allocation: %s", prev_allocations)  # 记录前一个分配

        # 初始化推理任务分配
        infer_nodes = set()
        infer_alloc = {}
        prev_train_alloc = {}
        for job, alloc in prev_allocations.items():
            if job not in infer_jobs:
                if 'infer' not in job:
                    prev_train_alloc[job] = alloc  # 记录前一个训练任务分配
                continue
            else:
                infer_alloc[job] = alloc  # 记录前一个推理任务分配

            for node_id in set(alloc):
                if self.node_id_dict[node_id] >= len(nodes) // 2:  # 推理节点的判断
                    infer_nodes.add(node_id)

        # 构建资源类型列表
        rtypes = sorted(set.union(*[set(job.resources) for job in self._jobs.values()]))
        # 构建任务资源矩阵
        self._job_resources = np.zeros((len(self._jobs), len(rtypes)), dtype=np.int64)
        for j, job in enumerate(self._jobs.values()):
            for r, rtype in enumerate(rtypes):
                self._job_resources[j, r] = job.resources.get(rtype, 0)

        # 构建节点资源矩阵
        self._node_resources = np.zeros((len(nodes), len(rtypes)), dtype=np.int64)
        for n, node in enumerate(nodes.values()):
            for r, rtype in enumerate(rtypes):
                self._node_resources[n, r] = node.resources.get(rtype, 0)

        # 计算每个任务的主导资源份额
        shares = self._job_resources / np.sum(self._node_resources, axis=0)
        self._dominant_share = np.amax(shares, axis=1)

        # 计算公平分配的副本数和节点数
        fair_replicas = np.ceil(1.0 / self._dominant_share / len(self._jobs))
        fair_nodes = np.ceil(len(nodes) * self._dominant_share)

        # 更新任务的速度提升函数
        for job, num_nodes, num_replicas in zip(self._jobs.values(), fair_nodes, fair_replicas):
            if not hasattr(job.speedup_fn, "_goodput_fn"):
                job.speedup_fn = lambda n, r: r / num_replicas
                continue

            job.speedup_fn._base_goodput = job.speedup_fn._goodput_fn.optimize(
                num_nodes=num_nodes, num_replicas=max(num_replicas, num_nodes),
                max_batch_size=job.speedup_fn._max_batch_size,
                atomic_bsz_range=job.speedup_fn._atomic_bsz_range,
                accumulation=job.speedup_fn._accumulation)[0]

        allocations = {}
        allocations.update(infer_alloc)  # 更新推理任务分配

        # 计算空闲 GPU 资源
        free_gpus = self.get_free_gpus(total_gpus, infer_alloc)#减去推理集群被占用的资源

        # 分配训练任务
        train_alloc = self.allocate_elastic(prev_train_alloc, self._jobs, free_gpus)
        remain_gpus = self.get_free_gpus(free_gpus, train_alloc)
        allocations.update(train_alloc)  # 更新训练任务分配

        # 分配睡眠任务
        sleep_alloc = {}
        for job in sleep_jobs:
            alloc = prev_allocations[job]
            if remain_gpus[alloc[0]] > 0:
                sleep_alloc[job] = alloc
                remain_gpus[alloc[0]] -= 1

        allocations.update(sleep_alloc)  # 更新睡眠任务分配

        return allocations, len(nodes)  # 返回最终分配和节点数量