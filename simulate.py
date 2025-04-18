"""
模拟器模板文件
用于模拟深度学习训练和推理任务的调度过程
"""
import argparse
import collections
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy
import glob
import json
import math
import pandas
import time
from old_application import APPLICATIONS, APPLICATIONS_DELAY, FIRST_DELAY, NEXT_DELAY
from goodput import GoodputFunction, fit_perf_params
from speedup import SpeedupFunction
from new_utils import JobInfo, NodeInfo
from aryl import Aryl
from dp import DeepBoot

LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
LOG.addHandler(ch)

# By default use ATS-I to schedule the inference task, otherwise use pollux's NSGA II
INFER_SCHEDULER = True

# By default True, which means if use NSGA II to schedule inference task, we make sure training allocation updates
# only when interval (60s)
REPAIR_TRAIN = True

# Inference schedule in Aryl
ARYL = True

# ATS-I to manage the lifecycle of inference task
AISS = True

# AFE to reduce the allocation update cost
AFE = False

PROTECT_TIMES = 1
RANDOM_ALLOCATE = 0

t_protect_max = 120
t_bonus = 15
t0 = 30
NUM_NODE = -1
schedule_cost_dict = {}
# 配置日志
LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)

def calculate_protect_time(info):
    t_protect_max = PROTECT_TIMES * 120
    t_bonus = PROTECT_TIMES * 15
    t0 = PROTECT_TIMES * 30
    return int(min(t0 + info.get('cache_times', 0) * t_bonus, t_protect_max))

def app_trans(app_name):
    app_name_split = app_name.split('-')
    # LOG.info("app name split: %s")
    if len(app_name_split) == 2 or 'infer' not in app_name:
        return app_name
    else:
        return "-".join(app_name_split[:-1])




class Job:
    """
        任务类，表示一个训练或推理任务
    """
    def __init__(self, name: str,  submission_time,application,
                 target_num_replicas = None, target_batch_size = None,
                 duration = None, required_gpu: int = 100):
        self.name = name
        self.application = application# 此处是application类
        self.submission_time = submission_time
        self.target_num_replicas = target_num_replicas#训练任务的动态副本数目标（弹性扩缩容）
        self.target_batch_size = target_batch_size#训练任务的批量大小（动态优化目标）。
        self.duration = duration#耗时，对训练任务无用
        self.required_gpu = required_gpu
        self.type = "training" if required_gpu == 100 else "inference"

        self.completion_time = None
        self.current_time = 0
        self.rescale_time = 0
        self.placement = ()
        self.atomic_bsz = None#原子批量大小（单个 GPU 一次处理的样本数）
        self.accum_steps = None#梯度累积步数（用于大批量训练）。
        self.profile = {}
        self.perf_params = None#性能模型参数（如计算时间、通信时间拟合值）
        self.grad_params = None#梯度统计参数（梯度方差、平方均值，用于效率计算）
        self.best_metric = None
        self.min_replicas = None  # 训练任务的最小副本数（弹性扩缩容下限）。
        self.max_replicas = None  # 训练任务的最大副本数（弹性扩缩容上限）。
        self.progress = 0.0
        self.epoch = 0
        self.attained_service = 0#累积服务时间（用于调度策略如 Pollux）
        self.num_restarts = None#训练任务的重启次数（弹性调度相关）
        self.inference = self.type == "inference"#冗余，可以看要不要去掉
        self.total_delay = 0
        self.total_delay_with_placement = 0
        self.start_execute_time = None
        self.evaluate_finish_time = None#预计完成时间
        self.run_time = 0
        self.current_rescale_time = 0
        self.protect_time = 30
        self.placement_update_history = []
        self.app = self.application.name
        if 'infer' in self.app:
            self.target_batch_size = 1
            self.target_num_replicas = 1
            self.inference = True
            self.atomic_bsz = 1
            self.pod_name = None
            self.status = "START"
            self.use_cache = False
    def get_goodput_fn(self):
        app = self.application
        return GoodputFunction(self.perf_params, self.grad_params, app.init_batch_size)

    def get_speedup_fn(self):
        if self.perf_params is None:
            return lambda n, r: r

        app = self.application
        return SpeedupFunction(self.get_goodput_fn(), app.max_batch_size,
                               (app.min_local_bsz, app.max_local_bsz),
                               accumulation=True)

    def update_local_bsz(self, placement):
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        if batch_size is None and self.perf_params is None:
            batch_size = max(app.init_batch_size,
                             app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn()
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, app.max_local_bsz), accumulation=True)
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            self.accum_steps = math.ceil(
                local_bsz / app.max_local_bsz - 1e-8) - 1
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(
                local_bsz / (self.accum_steps + 1) - 1e-8)
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))

    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        self.grad_params = (grad_sqr, grad_var)
        if (num_nodes, num_replicas, local_bsz) in self.profile:
            return
        self.profile[num_nodes, num_replicas, local_bsz] = step_time, sync_time
        num_nodes = np.array([key[0] for key in self.profile])
        num_replicas = np.array([key[1] for key in self.profile])
        local_bsz = np.array([key[2] for key in self.profile])
        step_time = np.array([val[0] for val in self.profile.values()])
        sync_time = np.array([val[1] for val in self.profile.values()])
        compute_time = step_time - sync_time
        self.perf_params = fit_perf_params(
            num_nodes, num_replicas, local_bsz, compute_time, step_time)

    def job_reallocate(self, placement):
        if not self.inference:#训练任务
            self.placement_update_history.append((self.current_time, self.placement, tuple(placement)))
        if placement:
            LOG.info("origin placement: %s, curr placement: %s",
                     self.placement, placement)
            origin_placement = self.placement
            self.placement = tuple(placement)#新的placement
            if not self.inference:#训练任务，要考虑重启次数
                self.update_local_bsz(self.placement)
                if AFE:
                    self.rescale_time = self.calculate_rescale_time(
                        origin_placement, placement)
                else:
                    delay_dict = APPLICATIONS_DELAY

                    # Start re-scale countdown. 这里要根据任务类型改
                    self.rescale_time = delay_dict[self.application.name]
                # self.rescale_time = 0 # 理论上限
                self.current_rescale_time = self.rescale_time

                if self.num_restarts is None:
                    self.num_restarts = 0
                else:
                    self.num_restarts += 1

            elif len(placement) > 0 and self.start_execute_time is None:#未开始执行的推理任务
                app_name = app_trans(self.app)
                self.rescale_time = APPLICATIONS_DELAY[app_name]

                self.start_execute_time = self.current_time
                self.evaluate_finish_time = self.start_execute_time + self.duration + self.rescale_time#启动时间+耗时

        else:  # placement为空，释放资源
            self.placement = ()
            self.atomic_bsz = 0


    def job_step(self, seconds, interference= 0.0, cluster=None):
        if not self.placement:
            if not self.inference or self.status == 'FINISH' or self.submission_time > cluster.clock:
                self.current_time += seconds
                return

        if self.inference:
            self._step_inference(seconds, cluster)
        else:
            self._step_training(seconds, interference)

    def _step_inference(self, seconds, cluster):
        infer_pod_status = cluster.infer_pod_status
        if not self.placement:
            if self.status == 'START' and len(infer_pod_status.get(self.app, {})) > 0:
                LOG.info("%s seeks for infer pod", self.name)
                LOG.info("infer pod status: %s", infer_pod_status)

                # 缓存次数高的pod优先执行
                infer_pod_status[self.app] = dict(sorted(infer_pod_status[self.app].items(),
                                                         key=lambda x: x[1].get('cache_times', 0), reverse=True))
                infer_pod_app = infer_pod_status[self.app]

                for pod_name, info in infer_pod_app.items():
                    '''
                    满足该条件则可以走推理缓存,不需要启动时间
                    '''
                    if info['status'] == 'SLEEP' or info['status'] == 'PROTECT' and AISS:
                        LOG.info("find the infer pod %s", pod_name)

                        self.status = 'RUNNING'
                        self.pod_name = pod_name
                        self.rescale_time = 0
                        self.use_cache = True
                        self.start_execute_time = cluster.clock
                        self.evaluate_finish_time = self.start_execute_time + self.duration  # 计算完成时间
                        self.completion_time = self.evaluate_finish_time
                        self.current_time = cluster.clock
                        self.total_delay = self.start_execute_time - self.submission_time
                        cluster.submit_time[self.evaluate_finish_time] = 1

                        if info['status'] == 'SLEEP':
                            info['cache_times'] = 0

                        else:
                            info['cache_times'] = info.get(
                                'cache_times', 0) + 1  # 缓存命中保护时间 + 1

                        info['curr_job'] = self.name
                        info['live_time'] = calculate_protect_time(info)
                        info['status'] = 'RUNNING'
                        info['completion_time'] = self.completion_time

                        return

                LOG.info("No PROTECT or SLEEP pod")
                self.current_time += seconds
                return

            else:  # 未完成的任务
                self.current_time += seconds
                return

        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        self.total_delay += delay
        self.total_delay_with_placement += delay * sum(self.placement)
        seconds -= delay

        while seconds > 0 and self.completion_time is None:
            assert self.epoch < self.application.max_epochs
            LOG.info("else infer: %s %s", self.name, self.pod_name)
            info = infer_pod_status[self.app][self.pod_name]
            self.completion_time = self.current_time + self.duration  # 此刻开始执行
            self.current_time += seconds

            flag = False
            if cluster.clock >= self.completion_time:

                info['status'] = 'PROTECT'
                info['curr_job'] = None
                flag = True

            else:
                info['status'] = 'RUNNING'
                info['curr_job'] = self.name

            info['live_time'] = calculate_protect_time(info)
            info['completion_time'] = self.completion_time

            if not AISS and flag:
                infer_pod_status[self.app].pop(self.pod_name)

            return
        self.current_time += seconds  # Add any remaining time.

    def _step_training(self, seconds, interference: float):
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        self.total_delay += delay
        self.total_delay_with_placement += delay * sum(self.placement)
        seconds -= delay

        while seconds > 0 and self.completion_time is None:
            assert self.epoch < self.application.max_epochs
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            num_nodes, num_replicas = len(placement), sum(placement)
            local_bsz = self.atomic_bsz
            batch_size = num_replicas * \
                         self.atomic_bsz * (self.accum_steps + 1)  # Pollux论文中的M
            scale = batch_size / self.application.init_batch_size
            # Calculate true (simulated) throughput.
            step_time, sync_time = \
                self.application.get_throughput(placement, self.atomic_bsz)
            accum_time = step_time - sync_time
            # Calculate true (simulated) efficiency.
            grad_sqr, grad_var = \
                self.application.get_grad_stats(batch_size, self.epoch)
            gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            total_time = step_time + accum_time * self.accum_steps
            goodput = gain / total_time * (1.0 - interference)
            # Update current epoch and progress.
            next_progress = self.application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                self.run_time += seconds
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                self.epoch += 1
                delta = round(
                    float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                completion_epoch = \
                    self.application.get_completion_epoch(batch_size)
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = \
                    self.application.get_best_metric(
                        batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                self.run_time += delta
                seconds -= delta
            # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement)
        self.current_time += seconds  # Add any remaining time.

class Cluster:
    """
    集群类，管理所有任务和资源
    """
    def __init__(self, workload, policy, min_nodes: int, num_gpus = 4,
                 max_nodes: Optional[int] = None, interference = 0.0,
                 low_util: Optional[float] = None, high_util: Optional[float] = None):
        """
        初始化集群
        :param workload: 工作负载
        :param policy: 调度策略
        :param min_nodes: 最小节点数
        :param num_gpus: 每个节点的GPU数量
        :param max_nodes: 最大节点数
        :param interference: 干扰系数
        :param low_util: 低利用率阈值
        :param high_util: 高利用率阈值
        """
        self.workload = workload
        self.policy = policy
        self.min_nodes = self.num_nodes = min_nodes
        self.num_gpus = num_gpus
        self.max_nodes = min_nodes if max_nodes is None else max_nodes
        self.interference = interference
        self.low_util = low_util
        self.high_util = high_util
        self.current_time = 0
        self.clock = 0
        self.jobs = []  # 所有任务列表

        self.allocations = {}  # 当前资源分配
        # self.infer_scheduler = InferScheduler()
        # self.infer_scheduler.aryl = ARYL

        self.infer_pod_status = dict()#TODO:修改逻辑
        self.protect_time = 30
        total_gpus = self.num_gpus * self.num_nodes
        self.optimize_history = []  # time, base_state.shape, cost

        for row in self.workload.itertuples():
            isinfer=app_trans(row.application).startswith("infer")
            if not (isinfer):#是训练任务
                self.train_jobs.append(Job(row.name, row.time,APPLICATIONS[app_trans(row.application)], row.num_replicas,row.batch_size))
            else:
                self.infer_jobs.append(
                    Job(row.name, row.time, APPLICATIONS[app_trans(row.application)],row.requested_gpu, duration=row.duration))


    def event_update(self):
        is_training_interval = self.clock % args.interval == 0  # 训练任务固定调度周期

        if is_training_interval:
            LOG.info("case1： 训练集群固定周期调度")
            return True

        # 有训练或推理任务完成
        is_job_finish = self.clock in self.submit_time and self.submit_time[self.clock] > 0

        if is_job_finish:
            LOG.info("case2：有任务完成")
            return True
        #考虑未分配资源的推理任务
        suspend_job_app = set()
        for job in self.jobs:
            if job.inference and job.completion_time is None and job.submission_time <= self.clock and len(
                    job.placement) == 0:
                suspend_job_app.add(job.app)

        if len(suspend_job_app) == 0:#没找到
            return False

        for app, pods in self.infer_pod_status.items():
            for name, info in pods.items():
                status = info['status']
                if status == 'SLEEP':
                    return True

                elif status == 'PROTECT' and app in suspend_job_app:
                    return True

        return False

    def aryl_remove(self, allocation):
        # LOG.info("alloc: %s",allocation)
        infer_nodes = set()
        for job, alloc in allocation.items():
            if 'infer' in job and len(alloc) > 0:
                infer_nodes.add(alloc[0])

        new_alloc = {job: [] for job in allocation}
        for job in new_alloc:
            if 'infer' in job:
                new_alloc[job] = allocation[job]
                continue

            for node in allocation[job]:
                if node not in infer_nodes:
                    new_alloc[job].append(node)

        return new_alloc
    def init_new_pod_status(self):
        for job in self.jobs:
            # job = self.job_dict[name]
            if not job.inference or job.name in self.finish_job_set or job.name not in self.allocations or len(self.allocations[job.name]) == 0:
                continue

            if job.app not in self.infer_pod_status:
                self.infer_pod_status[job.app] = dict()

            if job.name not in self.infer_pod_status[job.app]:
                self.infer_pod_status[job.app][job.name] = {
                    'curr_job': job.name,
                    'status': 'RUNNING',
                    'live_time': job.protect_time,
                    'completion_time': np.inf
                }
                job.pod_name = job.name

    def cluster_step(self, seconds: int = 60, interval: int = 60):
        """
        集群执行一个时间步
        :param seconds: 时间步长
        :param interval: 训练集群策略的调度间隔
        """
        #找出运行了多个任务的节点的索引，用于后续计算干扰系数
        interfere_nodes = set(idx for idx in range(self.num_nodes)
                              if sum(len(set(val)) > 1 and idx in val
                                     for key, val in self.allocations.items()) > 1)
        for job in self.jobs:
            job.clock = self.clock
            if job.completion_time and job.completion_time <= self.clock:
                job.status = 'FINISH'
            alloc_set = set(self.allocations.get(job.name, []))

            interference = 0.0#干扰系数
            if len(alloc_set) > 1 and any(idx in interfere_nodes for idx in alloc_set):
                interference = self.interference

            job.step(seconds, interference=interference, cluster=self)#每个任务推进相应时间

            if job.completion_time and job.name not in self.finish_job_set:#记录新完成的任务
                # finish_job_list.append(job)
                self.finish_job_set.add(job.name)
                LOG.info("finish job set %s", self.finish_job_set)

        self.current_time += seconds#更新集群当前时间
        LOG.info("cluster current time: %s", self.current_time)
        assert all(job.current_time == self.current_time for job in self.jobs)
        job_infos = self.get_job_infos()
        if job_infos:
            #判断是否需要自动伸缩，暂时略过
            # Optimize allocations.
            #先获取节点信息和分配方案信息
            node_infos = self.get_node_infos()
            #分别获取训练跟推理任务的allocation
            infer_allocations = {}
            training_allocations = {}

            for job_name, alloc in self.allocations.items():
                if job_name in job_infos:
                    if job_infos[job_name].inference:
                        infer_allocations[job_name] = alloc
                    else:
                        training_allocations[job_name] = alloc

            t1 = time.time()#优化前时间戳
            '''
            if INFER_SCHEDULER: #启用推理集群策略
                if self.clock % interval == 0:#训练集群策略
                    results = self.policy.optimize(
                        job_infos,
                        node_infos,
                        self.allocations,
                        node_infos[0],
                        self.clock,
                        self.infer_pod_status,
                    )#nodes_infos[0]未使用

                else:#推理集群策略
                    results = self.infer_scheduler.optimize(
                        job_infos,
                        node_infos,
                        self.allocations,
                        node_infos[0],
                        self.infer_pod_status,
                    )

            else:#未启用推理集群策略，只用训练集群策略
                results = self.policy.optimize(
                    job_infos,
                    node_infos,
                    self.allocations,
                    node_infos[0],
                    self.clock,
                    self.infer_pod_status
                )
            '''
            results=self.policy.optimize()
            t2 = time.time()#优化结束的时间戳
            optimize_time = round(t2 - t1, 3)#精确到3位小数
            #以下这段暂时不知道何用意
            num_jobs = len(job_infos)
            num_nodes = len(node_infos)

            if num_jobs not in schedule_cost_dict:
                schedule_cost_dict[num_jobs] = []

            schedule_cost_dict[num_jobs].append(
                {'cost': optimize_time, 'clock': self.clock, 'nodes': num_nodes}
            )

            allocations, _ = results#优化调度的结果，_为省去第二个参数len(nodes)
            if ARYL:
                allocations = self.aryl_remove(allocations)#如果训练任务占用了推理任务的资源，把这个资源去掉，还给推理任务
            #打印日志
            LOG.info("allocations: %s", allocations)
            LOG.info("optimize time: %s", optimize_time)
            LOG.info("schedule_cost_dict: %s", {
                'cost': optimize_time, 'clock': self.clock, 'nodes': num_nodes})
            # LOG.info("infer pod status: %s",self.infer_pod_status)
            #疑似状态更新
            for job, alloc in allocations.items():
                job_application = self.job_dict[job].app
                if job_application in self.infer_pod_status and job in self.infer_pod_status[job_application]:
                    pod_info = self.infer_pod_status[job_application][job]
                    if pod_info['status'] == 'SLEEP' and len(alloc) == 0:
                        LOG.info("pop %s", job)
                        self.infer_pod_status[job_application].pop(job)

            states = self._allocations_to_state(allocations, job_infos, node_infos)#获得状态矩阵
            self.optimize_history.append((self.clock, states.shape, optimize_time))#添加到历史记录
            #断言检查
            used_gpus = collections.Counter(sum(allocations.values(), []))
            assert all(val <= node_infos[key].resources["nvidia.com/gpu"]
                       for key, val in used_gpus.items())
            #从allocation获取placement
            for job in self.jobs:
                if allocations.get(job.name) != self.allocations.get(job.name):
                    alloc = allocations.get(job.name, [])
                    job.alloc = alloc
                    placement = []
                    for i in range(len(alloc)):
                        if i == 0 or alloc[i] != alloc[i - 1]:
                            placement.append(1)
                        else:
                            placement[-1] += 1
                    job.job_reallocate(placement)
                    if job.evaluate_finish_time and job.evaluate_finish_time not in self.submit_time:
                        self.submit_time[job.evaluate_finish_time] = 1

            self.allocations = allocations#更新分配
            self.init_new_pod_status()#初始化新的容器状态

        #记录日志
        self.current_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "optimize_history": self.optimize_history,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "allocation": self.allocations.get(job.name, []),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * sum(job.placement),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "completion_time": job.completion_time,
                    "grad_params": job.grad_params,
                    "rescale_time": job.rescale_time,
                    "run_time": job.run_time,
                    "start_execute_time": job.start_execute_time,
                    "evaluate_finish_time": job.evaluate_finish_time,
                    "delay_time": job.total_delay,
                    "placement_update_history": job.placement_update_history
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        }

    def get_job_infos(self,job) -> Dict:
        """
        获取所有任务的信息
        :return: 任务信息字典
        """
        job_info = JobInfo(
            job=job,
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            run_time=job.run_time,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),

            # max_replicas=min(64,  # simulator can't handle more.
            #                  job.application.max_batch_size // job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    #暂时不会触发
    def autoscale(self, job_infos):#使用二分查找算法在 min_nodes 和 max_nodes 之间寻找最佳节点数量，使得集群的利用率接近 target_utility。
        target_utility = (self.low_util + self.high_util) / 2
        min_nodes = self.min_nodes
        max_nodes = self.max_nodes
        num_nodes = self.num_nodes
        while min_nodes + 1 < max_nodes:
            utility = self.get_utility(num_nodes, job_infos)
            if utility < target_utility:
                max_nodes = num_nodes
            elif utility > target_utility:
                min_nodes = num_nodes
            else:
                break
            num_nodes = (min_nodes + max_nodes) // 2
        min_util = self.get_utility(min_nodes, job_infos)
        max_util = self.get_utility(max_nodes, job_infos)
        if abs(target_utility - min_util) < abs(target_utility - max_util):
            self.num_nodes = min_nodes
        else:
            self.num_nodes = max_nodes

    def get_node_infos(self, num_nodes=None):#原逻辑，键为节点下标，值为NodeInfo类对象
        return {
            idx: NodeInfo({"nvidia.com/gpu": self.num_gpus}, preemptible=False)
            for idx in range(num_nodes or self.num_nodes)
        }
        
    def all_complete(self):
        """
        检查所有任务是否完成
        :return: 是否所有任务都完成
        """
        return all(job.completion_time is not None for job in self.jobs)

def simulate(args):
    """
    模拟主函数
    :param args: 命令行参数
    :return: 模拟结果
    """
    workload="Workload/test.csv"
    traces="Trace"
    workload = pandas.read_csv(workload)
    policy = DeepBoot()#分配训练任务策略
    simulator = Cluster(workload, policy, args.min_nodes, num_gpus=args.num_gpus,
                        max_nodes=args.max_nodes, interference=args.interference,
                        low_util=args.low_util, high_util=args.high_util)
    simulator.protect_time = args.protect_time
    previous_clock = 0#记录之前上一次集群发生变化的时间

    while not (simulator.all_complete()):
        simulator.clock+=1
        #TODO：计算GPU利用率并输出日志

        #TODO：更新推理集群状态

        if not simulator.event_update():#没有状态变化则跳过
            continue
        #记录此次更新情况
        #LOG.info("+++ Clock: %s, real_gpu_use: %s, gpu_use: %s +++",
        #         simulator.clock, real_gpu_util, gpu_util)

        interval = simulator.clock - previous_clock#两次变化中间经过的时间单位
        LOG.info("previous: %s", previous_clock)
        LOG.info("clock: %s", simulator.clock)
        LOG.info("interval: %s", interval)
        #调用步进函数
        simulator.cluster_step(interval, args.interval)
        LOG.info("infer pod status: %s", simulator.infer_pod_status)

        '''
        LOG.info("infer pods: %s", infer_pods)
        LOG.info("---------------- SIMULATOR TIME: {} ----------------"
                 .format(simulator.current_time))
        LOG.info("Active jobs:")
        # for val in simulator.logs[-1]["submitted_jobs"]:
        for val in simulator.current_log['submitted_jobs']:
            if val["submission_time"] <= simulator.current_time and (
                    val["completion_time"] is None or val['name'] in infer_pods):
                LOG.info(
                    "    {}:\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}] \t[rescale time] {} \t[start execute time {}] \t[evaluation finish time {}] \t[total delay time {}] \t[completion time {}]".format(
                        val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"],
                        val["rescale_time"], val["start_execute_time"], val["evaluate_finish_time"], val["delay_time"],
                        val['completion_time']))
        used_gpus = sum(map(len, simulator.allocations.values()))
        LOG.info("GPU utilization: {}".format(used_gpus))
        LOG.info("Completed jobs:")
        jct_dict = simulator.get_jcts()
        LOG.info(jct_dict)

        LOG.info("Average JCT: %s", sum(jct_dict.values()) /
                                    len(jct_dict) if jct_dict else 0)
        '''
        previous_clock = simulator.clock#更新时间


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser()
    # 添加参数...
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str,
                        default="Workload/test.csv")
    parser.add_argument("--policy", type=str, default="dp",
                        choices=["tiresias", "optimus", "pollux", "afs", "aryl", 'dp'])
    parser.add_argument("--min-nodes", type=int, default=4,
                        help="min number of nodes in the cluster")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="max number of nodes for cluster autoscaling")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--infer_priority", type=int, default=1,
                        help="infer job has higher priority than training job")
    parser.add_argument("--protect_time", type=int, default=30,
                        help="protect time for inference replicas")
    parser.add_argument("--interference", type=float, default=0.0,
                        help="job slowdown due to interference")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="number of GPUs per node")

    parser.add_argument("--ARYL", type=int, default=0,
                        help="whether aryl schedule")

    parser.add_argument("--AISS", type=int, default=1,
                        help="whether aiss lifesycle")

    parser.add_argument("--AFE", type=int, default=0,
                        help="whether AFE optimize elastic")

    parser.add_argument("--INFER_SCHEDULER", type=int, default=1,
                        help="1 means using Pollux to schedule Inference tassks")

    parser.add_argument("--REPAIR_TRAIN", type=int, default=1,
                        help="1 means training task can't expand unless in interval")

    parser.add_argument("--protect_times", type=float, default=1.0,
                        help="1 means using Pollux to schedule Inference tasks")

    parser.add_argument("--random_allocate", type=int,
                        default=0, help="inference task random allocate")

    parser.add_argument("--log_file", type=int, default=0,
                        help="log out")

    parser.add_argument("--low-util", type=float,
                        help="low utility threshold")
    parser.add_argument("--high-util", type=float,
                        help="high utility threshold")
    parser.add_argument("--output", type=str, default="result",
                        help="path to output logs")
    parser.add_argument("--gpu_output", type=str,
                        help="path to output gpu usage info")
    parser.add_argument("--metric_output", type=str,
                        help="path to output metric info")
    args = parser.parse_args()
    
    # 运行模拟
    simulate(args) 