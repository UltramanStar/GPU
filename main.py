#场景：一个集群中有16个节点，每个节点各4个GPU，集群的GPU全部相同。前8个节点构成训练集群，后8个节点构成推理集群
#训练任务和推理任务随机到来，数据集为test.csv，inference开头的表示推理任务。
# 其中requested_gpu列表示一个副本需要的GPU空间占比。训练任务全部需要整个GPU，即GPU独占任务。
#推理任务需要的GPU小于一个，例如requested_gpu=50表示需要0.5个GPU。
#推理集群的GPU允许同时运行多个推理任务，训练集群的GPU只能同时允许单个训练任务。
#在仿真器中有三个类，GPU、Job、Cluster。GPU记录了自己的id，所属节点的id、当前的剩余可用空间、正在执行的任务列表和是否支持GPU共享
#Job类记录了任务名、应用类型、是否推理任务和分配情况。分配情况的格式是"name":[GPU]
# Cluster中包含了任务列表，所有GPU的状态和仿真器运行的方法
#每隔10个时间单位进行一次调度，训练集群和推理集群分别采取对应的策略
#推理集群的GPU需要进行状态管理，有任务运行时（即使未被全部使用）状态为RUNNING，所有任务运行结束后为PROTECT
#新到来的推理任务优先分配给PROTECT状态的GPU，如果在PROTECT_TIME内（默认为30个时间单位）没有被分配任务，状态变为FREE
#推理任务和训练任务均允许排队，处理策略为先到先服务。训练任务的总耗时为num_replicas*60个时间单位。例如num_replicas为4，
#但训练集群只剩2个可用GPU，则依然可以分配，但耗时翻倍。

import argparse
import csv
import os
import json
import time
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from models import GPU, Job_info
from goodput import GoodputFunction, fit_perf_params
from speedup import SpeedupFunction
from Random import Random
from firstfit import FirstFit
from Cache import CacheFirst
from BestFit import BestFit
from new_dp import DeepBoot
from application import APPLICATIONS, APPLICATIONS_DELAY, FIRST_DELAY, NEXT_DELAY
import logging

LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ch = logging.StreamHandler()
# ch.setFormatter(formatter)
# LOG.addHandler(ch)
#一些策略选择的参数
AFE=True
PROTECT_TIMES = 1

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
    def __init__(self, name: str, time: int, application: str, num_replicas=1,
                 requested_gpu=7, batch_size=None, duration=0, num_task=0):
        self.name = name
        self.submit_time = time#提交时间
        self.application = APPLICATIONS[app_trans(application)]
        self.app=self.application.name
        self.target_num_replicas = num_replicas#数据中传入的目标副本数
        self.requested_gpu = requested_gpu
        self.target_batch_size = batch_size#动态优化的批量大小，数据集会提供初始大小
        self.duration = duration
        self.num_task = num_task#推理请求数，暂不做处理
        self.is_inference = name.startswith('inference')#是否是推理任务
        self.status="START"
        self.current_time=0#当前时间，与集群一致
        self.allocation = []  # 分配的GPU下标列表

        self.start_execute_time = None#任务实际分配到资源的时间，于第一次分配到资源时更新
        self.evaluate_finish_time = None# 记录任务完成时间,主要是推理任务
        self.completion_time=None#判断训练任务是否完成。

        self.node_gpu_distribution = []  # 记录每个节点分配的GPU数量，暂无实际作用
        self.placement = []  # 只记录非零的GPU分配情况
        self.rescale_time=0
        #训练任务相关参数
        self.atomic_bsz = 0#原子批量大小（单个 GPU 一次处理的样本数）
        self.accum_steps = 0#梯度累积步数（用于大批量训练）。
        self.profile = {}
        self.max_epoch = 0  # 最大训练轮数
        self.epoch = 0  # 当前训练轮数
        self.perf_params = None#性能模型参数（如计算时间、通信时间拟合值）
        self.grad_params = None#梯度统计参数（梯度方差、平方均值，用于效率计算）
        self.best_metric = None
        self.progress = 0.0
        self.attained_service = 0#累积服务时间（用于部分调度策略）
        self.num_restarts = None#训练任务的重启次数（弹性调度相关）
        self.placement_update_history=[]#placement的历史变化情况
        self.total_delay = 0#对于推理任务为排队时长；对于训练任务为？
        self.total_delay_with_placement = 0
        self.use_cache=False
        self.run_time = 0
        self.current_rescale_time = 0

        if self.is_inference:
            self.target_batch_size = 1
            self.target_num_replicas = 1
            self.atomic_bsz = 1





    #一系列训练任务相关的方法
    @property
    def max_profiled_replicas(self):
        # temp=max((k[1] for k in self.profile), default=0)
        # if temp >= 12:
        #     print(f"{self.name}的max_replicas:{temp}")
        return max((k[1] for k in self.profile), default=0)

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

    def update_local_bsz(self, placement):#第一次计算吞吐量前，肯定先更新了self.atomic_bsz
        app = self.application
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)#当前的分配情况

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
        #print(f"{self.name}的profile{self.profile}")
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

    def job_allocate(self, placement):#传参为新的placement，目前只在这里设置推理任务的开始时间
        if not self.is_inference:#训练任务，将新的placement加入
            self.placement_update_history.append(
                (self.current_time, self.placement, tuple(placement))
            )

        if placement:
            LOG.info("origin placement: %s, curr placement: %s",
                     self.placement, placement)
            origin_placement = self.placement#原placement
            self.placement = tuple(placement)#更新placement
            if not self.is_inference:#训练任务
                if self.start_execute_time is None:
                    self.start_execute_time = self.current_time
                self.update_local_bsz(self.placement)#此处会更新self.atomic_bsz
                #获取重伸缩时长
                if AFE:
                    self.rescale_time = self.calculate_rescale_time(
                        origin_placement, placement)
                else:
                    delay_dict = APPLICATIONS_DELAY

                    # Start re-scale countdown. 这里要根据任务类型改
                    self.rescale_time = delay_dict[self.application.name]
                # self.rescale_time = 0 # 理论上限
                self.current_rescale_time = self.rescale_time
                #记录重启次数
                if self.num_restarts is None:
                    self.num_restarts = 0
                else:
                    self.num_restarts += 1
            #推理任务
            elif len(placement) > 0 and self.start_execute_time is None:

                app_name = app_trans(self.app)
                if self.use_cache:#有缓存

                    self.rescale_time = 0#直接使用缓存，不需要启动时间
                else:
                    self.rescale_time = APPLICATIONS_DELAY[app_name]

                self.start_execute_time = self.current_time

                self.evaluate_finish_time = self.start_execute_time + \
                    self.duration + self.rescale_time


        else:  # De-allocate all resources.新placement为空，释放资源
            print(f"{self.name}无placement调用job allocate")
            self.placement = ()
            self.atomic_bsz = 0

    def job_step(self, seconds,cluster=None,interference=0.0):

        if not self.placement:#没分配到资源，推理集群的处理逻辑
            if not self.is_inference or self.submit_time > cluster.clock or self.status == 'FINISH':#训练任务或者还未提交或者已完成
                self.current_time += seconds
                return  # 任务未开始或已结束，不需要更新
            elif self.is_inference and self.start_execute_time is not None:

                self.completion_time=self.evaluate_finish_time#目前不保留这行会有推理任务完成时间为None,因为缺少推理任务状态管理
                self.current_time += seconds
                return

        #接下来处理job有placement的情况

        delay = min(self.rescale_time, seconds)#计算延迟，对于推理任务为0
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        self.total_delay += delay
        self.total_delay_with_placement += delay * sum(self.placement)
        seconds -= delay

        while seconds > 0 and self.completion_time is None:
            # 更新训练任务的进度

            if not self.is_inference:#
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
                        self.application.get_completion_epoch(batch_size)#训练任务完成标准
                    if self.epoch > completion_epoch:#此处判断任务是否完成
                        self.completion_time = self.current_time + delta#训练任务完成
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
            else:#推理任务

                self.current_time += seconds

                return #退出循环
        self.current_time+=seconds #把剩下的seconds加上

    def calculate_rescale_time(self, origin_placement, current_placement):#AFE策略会用到
        app_name = self.application.name

        first_delay = FIRST_DELAY[app_name]
        next_delay = NEXT_DELAY[app_name]
        if len(origin_placement) == 0:  # First start, do not avoid the reduce cost
            return first_delay

        elif len(current_placement) == 0:
            return 0

        # 这里实际上是因为折算了
        if sum(origin_placement) < sum(current_placement):  # 扩容, 因为新重启的容器还要经历一次完整的重启时间 客观起见去掉
            return next_delay + (first_delay - next_delay) * (sum(current_placement) - sum(origin_placement)) // sum(current_placement)

        elif sum(origin_placement) > sum(current_placement):
            return next_delay

        else:  # 同扩同删的情况
            return first_delay

    def calculate_real_rescale_time(self, origin_placement, current_placement):#疑似没用
        app_name = self.application.name
        first_delay = FIRST_DELAY[app_name]
        next_delay = NEXT_DELAY[app_name]
        if len(origin_placement) == 0:
            return first_delay
        elif len(current_placement) == 0:
            return 0

        if sum(origin_placement) != sum(current_placement):
            return next_delay

        else:
            return first_delay
class Cluster:
    def __init__(self,infer_policy, num_nodes=16,max_nodes=None, num_gpus=4, interference=0,protect_time=0,cache_time=0,
                 SLO_time=15,low_util=None, high_util=None):

        self.jobs = []
        self.gpus = []
        self.num_nodes = num_nodes  # 集群的总节点数
        self.min_nodes=num_nodes
        self.max_nodes=num_nodes
        self.num_gpus = num_gpus  # 每个节点的GPU数量
        self.current_time = 0  # 集群的当前时间，用于step函数
        self.clock=0# 模拟的时间进程，用于simulate函数计数
        self.protect_time = protect_time  # PROTECT状态持续时间
        self.finished_jobs = set()  # 记录已完成的任务
        self.finish_job_set = set()  # 记录已完成的任务名
        self.interference = interference
        self.SLO_time=SLO_time
        self.low_util = low_util  # 利用率下界
        self.high_util = high_util  # 利用率上界
        self.allocations = {}  # 记录所有任务的分配情况，键为任务名，值为GPU下标列表
        self.jobs_submit_time = set()
        self.waiting_job={}
        self.jobs_finish_time=set()#记录完成时间的集合
        # 初始化策略
        self.preemptible=False#推理任务排队过长时，是否强制回收借出的GPU
        self.train_policy = DeepBoot()
        self.realtime=True#推理任务到来时，是即时处理还是定时处理
        self.infer_all_finish=False
        self.infer_policy = infer_policy


        #日志、指标记录相关
        self.logs = []
        self.optimize_history=[]
        self.current_log = []
        self.gpu_util_dict = {"clock": [], "real_gpu_use": [], "real_running_gpu_use": [], "gpu_use": []}


        self.gpu_usage_dict = {"clock": [], "usage": [
        ], "frag_ratio": [], "wasted_ratio": []}

        self.metric_dict = {
            "clock": [],
            "sum_goodput": [],
            "avg_goodput": [],
            "sum_speedup": [],
            "avg_speedup": []
        }
        self.fragmentation_dict={"clock": [],"fragementation":[]}#碎片化程度随时间变化的情况
        self.schedule_cost=0#调度算法花费的总时长

        # 初始化训练集群 (前半数个节点，每个节点4个GPU),GPU的id一直顺延
        gpu_id=0
        for node_id in range(num_nodes//2):
            for k in range(num_gpus):
                self.gpus.append(GPU(gpu_id, node_id, False,protect_time,cache_time))
                gpu_id+=1

        # 初始化推理集群 (后半数个节点，每个节点4个GPU)
        for node_id in range(num_nodes//2, num_nodes):  # 左闭右开
            for k in range(num_gpus):
                self.gpus.append(GPU(gpu_id, node_id, True,protect_time,cache_time))
                gpu_id += 1

    def print_cluster_status(self):
        """打印集群中每个GPU的状态"""


#一系列指标计算函数
    def get_jcts(self):#获取所有任务的JCT
        return {
            val["name"]: val["completion_time"] - val["submit_time"]
            # for val in self.logs[-1]["submitted_jobs"]
            for val in self.current_log["submitted_jobs"]
            if val["completion_time"] is not None
        }

    def get_infer_queueing_time(self):
        return {
            val["name"]: val["start_execute_time"] - val["submit_time"]
            # for val in self.logs[-1]["submitted_jobs"]
            for val in self.current_log["submitted_jobs"]
            if val["start_execute_time"] is not None and val["name"].startswith("inference")
        }

    def calculate_real_gpu_usage(self):
        real_used_gpu = 0
        used_gpu = 0
        real_running_gpu = 0
        #待修改计算方式

        for job in self.jobs:
            if job.submit_time <= self.current_time and job.completion_time is None:
                # 是训练任务:

                if job.current_rescale_time == 0:
                    real_used_gpu += sum(job.placement)

                    if not job.is_inference:
                        real_running_gpu += sum(job.placement)
                used_gpu += sum(job.placement)

            job.current_rescale_time = max(job.current_rescale_time - 1, 0)

        return real_used_gpu, real_running_gpu, used_gpu
    def calculate_goodput_and_speedup(self):
        speedups = []
        goodputs = []
        #job_infos = self.getDeepBootJobInfo()
        for job in self.jobs:
            if job.is_inference or job.name not in self.allocations:
                continue
            # job.name
            if job.submit_time <= self.current_time and job.completion_time is None:

                if job.grad_params is None or job.perf_params is None:
                    continue
                #job_info = job_infos[job.name]

                job_alloc = self.allocations[job.name]

                num_replicas = len(job_alloc)
                num_nodes =max (self.count_nodes(job_alloc),1)

                #goodput = job_info.speedup_fn._base_goodput
                goodput = job.get_speedup_fn()._base_goodput

                goodputs.append(goodput)

                #if not hasattr(job_info.speedup_fn, "_goodput_fn"):
                if not hasattr(job.get_speedup_fn(), "_goodput_fn"):
                    speedup_fn = lambda n, r: r / num_replicas
                else:
                    # print("has speedup_fn")
                    #speedup_fn = job_info.speedup_fn
                    speedup_fn = job.get_speedup_fn()

                speedup = speedup_fn(num_nodes, num_replicas)
                speedups.append(speedup)

                # print("goodput:",goodput)
                # print("speedup:",speedup)

        sum_goodput = 0
        avg_goodput = 0
        sum_speedup = 0
        avg_speedup = 0

        if len(goodputs) != 0:
            sum_goodput = round(np.sum(goodputs), 2)
            avg_goodput = round(np.average(goodputs), 2)

        if len(speedups) != 0:
            sum_speedup = round(np.sum(speedups), 2)
            avg_speedup = round(np.average(speedups), 2)

        return sum_goodput, avg_goodput, sum_speedup, avg_speedup

    def all_complete(self) -> bool:
        """检查所有任务是否都已完成"""

        return all(job.completion_time is not None for job in self.jobs)#所有任务都已完成


    def load_jobs(self, csv_file: str):
        df = pd.read_csv(csv_file)
        for row in df.itertuples():
            # job = Job(
            #     name=row.name,
            #     time=row.time,
            #     application=row.application,
            #     num_replicas=row.num_replicas,
            #     requested_gpu=row.requested_gpu,
            #     target_batch_size=row.batch_size,
            #     duration=row.duration,
            #     num_task=row.num_task
            # )
            job = Job(
                name=row.name,
                time=row.time,
                application=row.application,

                requested_gpu=row.requested_gpu,

                duration=row.duration,

            )
            if job.application.name == "ncf":
                job.target_batch_size = 32768
            self.jobs.append(job)
            if job.is_inference:
                self.jobs_submit_time.add(job.submit_time)


    def is_valid_job(self,job):
        if self.current_time >= job.submit_time and job.completion_time is None:#已提交未完成的任务
            return True
        if not job.is_inference:#训练任务不满足以上条件的为未提交或已完成
            return False
        return False#推理任务待定

    def getJobInfo(self):#返回jobinfo类对象的字典
        job_infos = []
        for job in self.jobs:

            if self.is_valid_job(job):
                job_info=Job_info(job,job.get_speedup_fn,job.submit_time,job.target_num_replicas,job.requested_gpu,
                                          job.duration,job.run_time)
                job_info.age=self.current_time-job.submit_time#部分策略有这个需求，未来优化判断逻辑

                job_info.num_restarts = job.num_restarts or 0

                job_infos.append(job_info)
        return job_infos
    def getDeepBootJobInfo(self):#返回jobinfo类对象的列表
        job_infos = []#TODO:确定job_infos是字典还是列表
        for job in self.jobs:

            if self.is_valid_job(job):

                job_info=Job_info(job=job,speedup_fn=job.get_speedup_fn(),submit_time=job.submit_time,attained_service=job.attained_service,
                                  num_replicas=job.target_num_replicas,
                                  max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),requested_gpu=job.requested_gpu,
                                          duration=job.duration,run_time=job.run_time)
                job_info.age=self.current_time-job.submit_time#部分策略有这个需求，未来优化判断逻辑
                job_info.num_restarts = job.num_restarts or 0
                if job.application.name == "ncf":
                    job_info.max_replicas = 1

                job_infos.append(job_info)
        return job_infos

    def get_train_gpus(self) -> List[GPU]:
        return [gpu for gpu in self.gpus if not gpu.is_inference]

    def get_idle_gpus(self) -> List[GPU]:
        gpu_list=[]
        for gpu in self.gpus:
            if not gpu.is_inference:#训练集群GPU
                gpu_list.append(gpu)

            elif gpu.state=="FREE" or gpu.state=="BORROWED":
                gpu_list.append(gpu)

        return gpu_list

    def get_infer_gpus(self) -> List[GPU]:
        return [gpu for gpu in self.gpus if gpu.is_inference]

    def count_nodes(self,alloc):#计算任务部署在多少个节点上。

        # 每个节点有4个GPU，所以节点数量是GPU数量除以4
        node_count = len(set(gpu // self.num_gpus for gpu in alloc))#表示这些GPU属于多少个不同的节点
        return node_count
    def need_update(self):#判断是否有更新的情况，有则需要调用step
        #训练任务固定调度周期


        is_training_interval = self.clock % args.train_interval == 0  # 训练任务固定调度周期
        is_infer_interval=self.clock % args.infer_interval == 0 # 推理任务固定调度周期
        if is_training_interval:
            LOG.info("case1:训练任务固定调度周期")
            return True
        #有推理任务到来或属于推理任务固定调度周期
        if self.clock in self.jobs_submit_time and self.realtime:
            LOG.info("case2:有推理任务提交")
            return True
        if is_infer_interval and not self.realtime and not self.infer_all_finish:
            LOG.info("case3:推理任务固定调度周期")
            return True
        #有任务完成
        if self.clock in self.jobs_finish_time:
            #print("case3:任务完成更新")
            LOG.info("case4:有推理任务完成")
            return True
        #有排队中的推理任务等待时间过长，尝试回收资源
        if self.waiting_job:
            LOG.info(f"当前时间{self.clock}，排队队列长度{len(self.waiting_job)}")
            if self.clock - max(self.waiting_job.values())>self.SLO_time:#有任务超过最大排队时间
                LOG.info("case5:任务排队时间过长")
                return True
        return False

    def update_cluster_states(self):  # simulate函数中每个时间步调用这个函数更新集群状态

        infer_gpus = self.get_infer_gpus()
        for gpu in infer_gpus:
            cache_to_remove = []#要删除的缓存
            for app,start_time in gpu.app_cache.items():
                if start_time == -1:#任务仍在执行
                    continue
                if self.clock>start_time+gpu.save_time:#缓存时间已过，记录要删除的缓存
                    cache_to_remove.append(app)
            for app in cache_to_remove:#遍历完再删除，不然形状变化会导致for循环无法继续
                gpu.app_cache.pop(app, None)

            if gpu.state == 'RUNNING':#有推理任务在运行
                for job in gpu.running_jobs:
                    if self.clock>job.evaluate_finish_time:#推理任务完成，释放
                        gpu.deallocate(job,job.requested_gpu,self.clock)


            if gpu.state=='PROTECT' and self.clock>gpu.protect_start_time+gpu.protect_time:#保留时间结束，释放预留资源
                #gpu.application_cache=set()#清空缓存应用
                gpu.state='FREE'


            #TODO：训练任务完成后，从BORROWED变成FREE，考虑是否保留缓存

    def update_infer_states(self):  # 更新完优化的结果后，调用这个函数来更新推理集群状态
        current_time = self.current_time
    def get_utility(self, num_nodes, job_infos, allocations=None):

        if allocations is None:
            # policy = copy.deepcopy(self.policy)
            allocations = self.train_policy.optimize(job_infos, self.allocations, self.gpus)

        sum_speedup = 0.0
        for key, alloc in allocations.items():
            if key in job_infos:
                speedup_fn = job_infos[key].speedup_fn
                speedup = speedup_fn(self.count_nodes(alloc), len(alloc))
                sum_speedup += speedup
        return sum_speedup / (num_nodes * self.num_gpus)

    def check_infer_gpu_usage(self):#返回存在碎片化情况的GPU个数，总共浪费的GPU空间百分比
        infer_gpus=self.get_infer_gpus()
        usage={}
        running_gpus=0  #非FREE状态的GPU
        fragment_gpus=0  #存在碎片化的GPU
        wasted_space=0  #总共浪费的空间
        frag_ratio = 0  #碎片化GPU占比
        wasted_ratio = 0  #浪费空间占比
        for gpu in infer_gpus:
            usage[gpu.gpu_id]=7-gpu.available_space
            if gpu.state != 'FREE':
                running_gpus+=1
                if gpu.state=='RUNNING':
                    if gpu.available_space>0:
                        fragment_gpus+=1
                        wasted_space+=7-gpu.available_space

        if running_gpus:
            frag_ratio=fragment_gpus/running_gpus
            wasted_ratio=wasted_space/(running_gpus*7)

        return usage,frag_ratio,wasted_ratio



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

    def apply_allocation(self, job: Job, alloc: List[int]):#此处不会释放已完成任务的资源
        """应用分配方案到集群"""

        origin=set(job.allocation)
        new=set(alloc)
        to_deallocate=list(origin-new)
        to_allocate=list(new-origin)


        if len(to_deallocate)>0:
            #print(f"释放任务 {job.name} 的原有资源:{to_deallocate}")
            for gpu_idx in to_deallocate:
                gpu = self.gpus[gpu_idx]
                if job in gpu.running_jobs:
                    gpu.deallocate(job, job.requested_gpu,self.current_time)

        for gpu_idx in to_allocate:
            gpu = self.gpus[gpu_idx]
            if not job.is_inference:
                gpu.running_jobs=[]
            gpu.allocate(job, job.requested_gpu,self.current_time)
        
        # 更新任务的allocation和集群的allocations
        job.allocation = alloc
        self.waiting_job.pop(job.name,None)
        job.status="RUNNING"#该逻辑会覆盖排队任务的WAIT状态


    def cluster_step(self, seconds: int, train_interval=60, infer_interval=10):
        #推进仿真进程seconds个时间单位
        #训练集群固定调度周期为train_interval
        print(f"\n=== 开始新的时间步 ===")
        print(f"当前集群时间: {self.current_time} 当前时钟: {self.clock}")

        #找出运行了多个任务的节点
        node_task_count = {}# 初始化一个字典来记录每个节点被多少个任务占用
        # 遍历每个任务的分配
        for job_name, gpu_list in self.allocations.items():
            # 提取该任务占用的节点下标
            nodes_occupied = set(gpu // self.num_gpus for gpu in gpu_list)
            # 更新每个节点的任务计数
            for node in nodes_occupied:
                if node not in node_task_count:
                    node_task_count[node] = 0
                node_task_count[node] += 1
        # 筛选同时运行了多个任务的节点
        interfere_nodes = set(node for node, count in node_task_count.items() if count > 1)


        for job in self.jobs:

            if job.evaluate_finish_time and self.clock>=job.evaluate_finish_time:
                job.completion_time=job.evaluate_finish_time
            if job.completion_time and job.completion_time <= self.clock:
                job.status = 'FINISH'
            alloc_set = set(gpu // self.num_gpus for gpu in self.allocations.get(job.name, []))
            interference = 0.0
            if len(alloc_set) > 1 and any(idx in interfere_nodes for idx in alloc_set):
                interference = self.interference

            job.job_step(seconds, self, interference=interference)
            if job.completion_time and job not in self.finished_jobs:
                if not job.is_inference:
                    for gpuID in job.allocation:
                        gpu = self.gpus[gpuID]
                        gpu.deallocate(job,job.requested_gpu,self.clock)
                job.allocation = []

                self.finished_jobs.add(job)
                self.finish_job_set.add(job.name)
                if not job.is_inference:
                    print(f"{job.name}应该完成了")
        LOG.info("已完成的任务集合：%s", self.finish_job_set)

        #job都step完以后进行优化
        self.current_time += seconds
        job_infos = self.getDeepBootJobInfo()
        current_job_names = [job.name for job in job_infos]

        if job_infos:
            if self.max_nodes > self.min_nodes:
                # Autoscale cluster if needed.
                self.utility.append(self.get_utility(
                    self.num_nodes, job_infos, self.allocations))
                if len(self.utility) > 15:
                    self.utility.pop(0)
                    utility = sum(self.utility) / len(self.utility)
                    if (self.num_nodes > self.min_nodes and utility < self.low_util) or \
                            (self.num_nodes < self.max_nodes and utility > self.high_util):
                        self.autoscale(job_infos)
                        self.utility.clear()
                    LOG.info("Utility: %s", utility)
                LOG.info("Nodes: %s", self.num_nodes)

            self.allocations = {
                k: v for k, v in self.allocations.items() if k in current_job_names}#未完成任务的allocation，不含新任务

            train_jobs = [job for job in job_infos if not job.is_inference]
            infer_jobs = [job for job in job_infos if job.is_inference]

            new_alloc={}
            t1 = time.time()

            if self.clock % train_interval == 0:#训练集群策略
                # 获取可用的训练GPU,以及FREE状态的推理GPU
                infer_schedule=False
                if self.realtime and self.clock in self.jobs_submit_time:
                    infer_schedule=True
                if not self.realtime and self.clock % infer_interval == 0:
                    infer_schedule = True
                if infer_schedule:#恰好也需要运行推理任务调度，则先调度推理任务
                    available_gpus = self.get_infer_gpus()
                    # 调用优化函数
                    new_alloc = self.infer_policy.optimize(job_infos, self.allocations, available_gpus,self.preemptible)

                    print("推理优化结果", new_alloc)

                available_gpus = self.get_idle_gpus()#含推理集群GPU

                # 调用优化函数
                infer_job_names = [job.name for job in infer_jobs]
                infer_alloc = {k: v for k, v in self.allocations.items() if k in infer_job_names}
                if not new_alloc:
                    new_alloc.update(infer_alloc)
                train_alloc = self.train_policy.optimize(job_infos, self.allocations, available_gpus)
                print("训练优化结果：",train_alloc)
                new_alloc.update(train_alloc)

                #print("训练优化结果", new_alloc)
            else:#有推理任务提交或完成，或推理任务固定调度周期
                # 获取可用的推理GPU
                available_gpus = self.get_infer_gpus()
                # 调用优化函数
                new_alloc = self.infer_policy.optimize(job_infos, self.allocations, available_gpus,self.preemptible)
                print("推理优化结果", new_alloc)
            #print("优化结果", new_alloc)
            t2=time.time()
            optimize_time=t2-t1
            self.schedule_cost+=optimize_time
            self.optimize_history.append((self.clock, optimize_time))#记录此次优化的耗时，原逻辑中还记录了形状
            LOG.info("allocations: %s", new_alloc)
            LOG.info("optimize time: %s", optimize_time)
            LOG.info("schedule_cost_dict: %s", {'cost': optimize_time, 'clock': self.clock})

            for job in self.jobs:
                if new_alloc.get(job.name) != self.allocations.get(job.name):#此处构建job的placement
                    alloc = new_alloc.get(job.name, [])
                    #job.alloc = alloc#意义不明
                    placement = []
                    if job.is_inference:
                        if len(alloc)>0:#推理任务要开始执行了
                            placement=[1]
                            #判断分配到的GPU上是否有缓存
                            gpu = self.gpus[alloc[0]]#前提：限定推理任务只在一个GPU上执行
                            if job.app in gpu.app_cache:
                                job.use_cache=True

                        else:#未分配到资源，开始排队
                            job.status='WAIT'
                            print(f"{job.name}未分配到资源，暂时排队")
                            if job.name not in self.waiting_job:
                                self.waiting_job[job.name]=self.current_time
                            continue
                    else:
                        # 计算训练任务的node_gpu_distribution
                        node_gpu_count = {}
                        for gpu_id in alloc:#此处为GPU的gpu_id
                            node_id=gpu_id//self.num_gpus#计算GPU对应的节点
                            if node_id not in node_gpu_count:
                                node_gpu_count[node_id] = 0
                            node_gpu_count[node_id] += 1
                        # 按节点ID排序并生成node_gpu_distribution列表）
                        job.node_gpu_distribution = [node_gpu_count.get(node_id, 0) for node_id in
                                                     range(self.num_nodes)]
                        placement = [count for count in job.node_gpu_distribution if count > 0]

                    #计算完placement布置任务
                    job.job_allocate(placement)#更新Job的开始、预计完成时间、placement
                    self.apply_allocation(job,alloc)
                    #记录任务的完成时间，在对应时间触发step

                    if job.evaluate_finish_time:
                        self.jobs_finish_time.add(job.evaluate_finish_time)

            self.allocations = new_alloc#更新集群的整体分配情况
            if self.jobs_submit_time and self.current_time>max(self.jobs_submit_time):
                self.infer_all_finish = True#所有推理任务均已部署，不用再定时调用推理集群策略

        # 更新推理GPU的状态
        #self.update_gpu_states()
        #self.update_infer_states()
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
                    "submit_time": job.submit_time,
                    "completion_time": job.completion_time,
                    "grad_params": job.grad_params,
                    "rescale_time": job.rescale_time,
                    "run_time": job.run_time,
                    "start_execute_time": job.start_execute_time,
                    "evaluate_finish_time": job.evaluate_finish_time,
                    "delay_time": job.total_delay,
                    "placement_update_history": job.placement_update_history
                }
                for job in self.jobs if job.submit_time <= self.current_time
            ],
        }



    #日志输出相关
    def output_logs(self, path):
        LOG.info("output_logs")
        if os.path.isdir(path):
            path = os.path.join(path, 'jobinfo.log')
        with open(path, "w") as f:
            # record = self.logs[-1]
            record = self.current_log
            json.dump(record, f)
            f.write("\n")


    def output_gpu_util_info(self, path):
        with open(path, "w") as f:
            record = self.gpu_util_dict
            # for record in self.logs:
            json.dump(record, f)

    def output_infer_usage_info(self, path):
        with open(path, "w") as f:
            record = self.gpu_usage_dict
            # for record in self.logs:
            json.dump(record, f)
    def output_metric_info(self, path):
        with open(path, "w") as f:
            record = self.metric_dict
            json.dump(record, f)
def simulate(args=None):
    # self.jobs.sort(key=lambda x: x.submit_time)

    # 处理传入的参数args

    # 根据infer_policy初始化对应的策略类
    if args.infer_policy == "Cache":
        policy = CacheFirst(args.SLO_time)
    elif args.infer_policy == "FirstFit":
        policy = FirstFit(args.SLO_time)
    elif args.infer_policy == "BestFit":
        policy = BestFit(args.SLO_time)
    else:
        policy = Random(args.SLO_time)
    simulator = Cluster(infer_policy=policy,num_gpus=args.num_gpus,interference=args.interference,max_nodes=args.max_nodes,
                        protect_time=args.protect_time,cache_time=args.cache_time,SLO_time=args.SLO_time,
                        low_util=args.low_util, high_util=args.high_util)
    simulator.protect_time = args.protect_time
    simulator.load_jobs(args.workload)
    simulator.preemptible=args.preemptible
    simulator.realtime=args.realtime
    previous_clock = 0  # 上一次调用step的时间
    while not simulator.all_complete():
        simulator.clock += 1
        #计算指标，添加日志记录
        real_gpu_util, real_running_gpu_util, gpu_util = simulator.calculate_real_gpu_usage()
        infer_gpu_usage,frag_ratio,wasted_ratio = simulator.check_infer_gpu_usage()
        #If calculate the goodput and speedup in the whole process, use follow code
        # sum_goodput, avg_goodput, sum_speedup, avg_speedup = simulator.calculate_goodput_and_speedup()
        #
        # simulator.metric_dict['clock'].append(simulator.clock)
        # simulator.metric_dict['sum_goodput'].append(sum_goodput)
        # simulator.metric_dict['avg_goodput'].append(avg_goodput)
        # simulator.metric_dict['sum_speedup'].append(sum_speedup)
        # simulator.metric_dict['avg_speedup'].append(avg_speedup)

        simulator.gpu_util_dict['clock'].append(simulator.clock)
        simulator.gpu_util_dict['real_gpu_use'].append(real_gpu_util)
        simulator.gpu_util_dict['real_running_gpu_use'].append(
            real_running_gpu_util)
        simulator.gpu_util_dict['gpu_use'].append(gpu_util)

        simulator.gpu_usage_dict['clock'].append(simulator.clock)
        #simulator.gpu_usage_dict['usage'].append(infer_gpu_usage)
        simulator.gpu_usage_dict['frag_ratio'].append(frag_ratio)
        simulator.gpu_usage_dict['wasted_ratio'].append(wasted_ratio)

        #TODO：目前在cluster_step中检查会导致最后一个任务的资源不释放，且没更新缓存状态
        simulator.update_cluster_states()
        if not simulator.need_update():
            continue
        interval = simulator.clock - previous_clock
        simulator.cluster_step(interval, args.train_interval,args.infer_interval)  # 推进interval个时间单位的模拟进程

        #记录日志
        LOG.info("---------------- SIMULATOR TIME: {} ----------------"
                 .format(simulator.current_time))
        LOG.info("Active jobs:")

        for val in simulator.current_log['submitted_jobs']:
            if val["submit_time"] <= simulator.current_time and (
                    val["completion_time"] is None ):#尚未考虑推理任务排队情形
                LOG.info(
                    "    {}:\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}] \t[rescale time] {} \t[start execute time {}] \t[evaluation finish time {}] \t[total delay time {}] \t[completion time {}]".format(
                        val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"],
                        val["rescale_time"], val["start_execute_time"], val["evaluate_finish_time"], val["delay_time"],
                        val['completion_time']))

        LOG.info("Completed jobs:")
        jct_dict = simulator.get_jcts()
        LOG.info(jct_dict)
        # 分别计算训练任务和推理任务的平均JCT
        train_jct_values = []
        infer_jct_values = []

        for name, jct in jct_dict.items():
            if name.startswith('inference'):
                infer_jct_values.append(jct)
            else:
                train_jct_values.append(jct)

        average_train_jct = sum(train_jct_values) / len(train_jct_values) if train_jct_values else 0
        average_infer_jct = sum(infer_jct_values) / len(infer_jct_values) if infer_jct_values else 0

        #LOG.info("Average JCT: %s", sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0)
        LOG.info("Average Train JCT: %s", average_train_jct)
        LOG.info("Average Infer JCT: %s", average_infer_jct)
        previous_clock = simulator.clock

    if args.output:#输出记录
        simulator.output_logs(args.output)
        simulator.output_gpu_util_info(args.gpu_output)
        simulator.output_metric_info(args.metric_output)
        simulator.output_infer_usage_info(args.usage_output)

    result_jcts = simulator.get_jcts()
    result_infer_queueing=simulator.get_infer_queueing_time()

    return simulator.logs, result_jcts, simulator.gpu_util_dict, simulator.metric_dict, simulator.gpu_usage_dict,\
           result_infer_queueing,simulator.schedule_cost

    #打印结果日志

if __name__ == "__main__":
    #输入args参数，待添加
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str,
                        default="Workload/newSmall.csv")
    parser.add_argument("--infer_policy", type=str, default="FirstFit",
                        choices=["FirstFit", "BestFit", "Cache", "Random"],
                        help="推理集群采用的策略")
    parser.add_argument("--min-nodes", type=int, default=16,
                        help="min number of nodes in the cluster")
    parser.add_argument("--max-nodes", type=int, default=None,
                        help="max number of nodes for cluster autoscaling")
    parser.add_argument("--train_interval", type=int, default=60,
                        help="训练集群固定调度周期")
    parser.add_argument("--preemptible", type=int, default=1,
                        help="是否强制回收")
    parser.add_argument("--protect_time", type=int, default=30,
                        help="GPU进入保护状态的时间")
    parser.add_argument("--cache_time", type=int, default=60,
                        help="缓存保留的时间")
    parser.add_argument("--interference", type=float, default=0.0,
                        help="干扰系数")
    parser.add_argument("--num-gpus", type=int, default=4,
                        help="number of GPUs per node")

    parser.add_argument("--ARYL", type=int, default=0,
                        help="whether aryl schedule")

    parser.add_argument("--realtime", type=int, default=1,
                        help="推理任务是否实时处理")

    parser.add_argument("--AFE", type=int, default=1,
                        help="whether AFE optimize elastic")

    parser.add_argument("--infer_interval", type=int, default=10,
                        help="推理集群每隔多久调用一次")

    parser.add_argument("--REPAIR_TRAIN", type=int, default=1,
                        help="1 means training task can't expand unless in interval")

    parser.add_argument("--protect_times", type=float, default=1.0,
                        help="1 means using Pollux to schedule Inference tasks")

    parser.add_argument("--SLO_time", type=int,default=10,
                        help="允许的最大排队时间")

    parser.add_argument("--log_file", type=int, default=0,
                        help="log out")

    parser.add_argument("--low-util", type=float,
                        help="low utility threshold")
    parser.add_argument("--high-util", type=float,
                        help="high utility threshold")
    parser.add_argument("--output", type=str, default="test",
                        help="path to output logs")
    parser.add_argument("--gpu_output", type=str,
                        help="path to output gpu usage info")
    parser.add_argument("--metric_output", type=str,
                        help="path to output metric info")
    parser.add_argument("--usage_output", type=str,
                        help="path to output 碎片化相关的信息")

    args = parser.parse_args()
    AFE = args.AFE

    REPAIR_TRAIN = args.REPAIR_TRAIN

    ARYL = args.ARYL
    PROTECT_TIMES = args.protect_times
    NUM_NODE = args.min_nodes

    #创建文件夹
    # exit()
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    files = os.listdir(args.output)
    for file in files:
        os.remove(os.path.join(args.output, file))

    if args.log_file:
        log_file = args.output + '/simulator.log'
        if os.path.exists(log_file):
            os.remove(log_file)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        LOG.addHandler(fh)

    LOG.info("output: %s", args.output)
    LOG.info("single workload")
    # exit()
    args.gpu_output = args.output + '/gpu.log'
    args.metric_output = args.output + '/metric.log'
    args.usage_output=args.output+'/infer.log'


    summary = {"jcts": {}, "avgs_train": {}, "avgs_infer":{},"schedule_cost":{}}
    inference_info={"queueing_time":{}, "avgs_time":{},"max_time":{}, "queue_tasks":{}, "queue_ratio":{}}
    logs, jct_dict, gpu_util_dict, metric_dict, infer_usage, infer_queueing,schedule_cost= simulate(args)#程序入口
    #JCT相关
    summary["jcts"] = jct_dict
    if len(jct_dict) == 0:
        summary["avgs_train"] = 0
    else:
        train_jct_values = []
        infer_jct_values = []

        for name, jct in jct_dict.items():
            if name.startswith('inference'):
                infer_jct_values.append(jct)
            else:
                train_jct_values.append(jct)

        avgs_train_jct = sum(train_jct_values) / len(train_jct_values) if train_jct_values else 0
        avgs_infer_jct = sum(infer_jct_values) / len(infer_jct_values) if infer_jct_values else 0
        summary["avgs_train"] = avgs_train_jct
        summary["avgs_infer"] = avgs_infer_jct
        summary["schedule_cost"]=schedule_cost

    #推理集群指标相关
    inference_info["queueing_time"]=infer_queueing
    inference_info["avgs_time"]=sum(inference_info["queueing_time"].values())/len(inference_info["queueing_time"])
    inference_info["max_time"] =max(inference_info["queueing_time"].values())
    num_queue=sum(1 for time in inference_info["queueing_time"].values() if time > 0)
    inference_info["queue_tasks"] = num_queue
    inference_info["queue_ratio"]=num_queue/ len(inference_info["queueing_time"]) if len(inference_info["queueing_time"])>0 else 0


    with open(args.output + "/summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    with open(args.output + "/inference.json", "w") as f:
        json.dump(inference_info, f, indent=4)
    #simulate()#程序入口






