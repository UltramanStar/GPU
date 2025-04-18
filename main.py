#场景：一个集群中有16个节点，每个节点各4个GPU，集群的GPU全部相同。前8个节点构成训练集群，后8个节点构成推理集群
#训练任务和推理任务随机到来，数据集为test.csv，inference开头的表示推理任务。
# 其中requested_gpu列表示一个副本需要的GPU空间占比。训练任务全部需要100%GPU，即GPU独占任务。
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

import csv
import time
import math
import pandas as pd
import numpy as np
from typing import List, Dict, Set
from models import GPU, JobInfo
from goodput import GoodputFunction, fit_perf_params
from speedup import SpeedupFunction
from train_policy import TrainPolicy
from infer_policy import InferPolicy
from application import APPLICATIONS, APPLICATIONS_DELAY, FIRST_DELAY, NEXT_DELAY
import logging

LOG = logging.getLogger('simulator')
LOG.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#一些策略选择的参数
AFE=False

def app_trans(app_name):
    app_name_split = app_name.split('-')
    # LOG.info("app name split: %s")
    if len(app_name_split) == 2 or 'infer' not in app_name:
        return app_name
    else:
        return "-".join(app_name_split[:-1])
        
class Job:
    def __init__(self, name: str, time: int, application: str, num_replicas: int, 
                 requested_gpu: int, batch_size: int, duration: int, num_task: int):
        self.name = name
        self.submit_time = time#提交时间
        self.application = APPLICATIONS[app_trans(application)]
        self.target_num_replicas = num_replicas#数据中传入的目标副本数
        self.requested_gpu = requested_gpu
        self.target_batch_size = batch_size#动态优化的批量大小，数据集会提供初始大小
        self.duration = duration
        self.num_task = num_task#推理请求数，暂不做处理
        self.is_inference = name.startswith('inference')#是否是推理任务
        self.status="START"#在原版中对推理任务有效，逻辑待定
        self.current_time=0#当前时间，与集群一致
        self.allocation = []  # 分配的GPU下标列表

        self.start_execute_time = None#任务实际分配到资源的时间，于第一次分配到资源时更新
        self.evaluate_finish_time = None# 记录任务完成时间,主要是推理任务
        self.completion_time=None#判断训练任务是否完成。TODO：将训练任务和推理任务的完成条件做区分
        #self.finish_time = None
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
        self.total_delay = 0
        self.total_delay_with_placement = 0

        self.run_time = 0
        self.current_rescale_time = 0

        if self.is_inference:
            self.target_batch_size = 1
            self.target_num_replicas = 1
            self.atomic_bsz = 1
            #self.pod_name = None
            #self.status = "START"
            self.use_cache = False



    #一系列训练任务相关的方法
    def max_profiled_replicas(self):
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

    def job_allocate(self, placement):#传参为新的placement
        if not self.inference:#训练任务，将新的placement加入
            self.placement_update_history.append(
                (self.current_time, self.placement, tuple(placement))
            )

        if placement:
            LOG.info("origin placement: %s, curr placement: %s",
                     self.placement, placement)
            origin_placement = self.placement#原placement
            self.placement = tuple(placement)#更新placement
            if not self.inference:#训练任务
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
                self.rescale_time = APPLICATIONS_DELAY[app_name]

                self.start_execute_time = self.current_time
                self.evaluate_finish_time = self.start_execute_time + \
                    self.duration + self.rescale_time


        else:  # De-allocate all resources.新placement为空，释放资源
            self.placement = ()
            self.atomic_bsz = 0

    def job_step(self, seconds,cluster=None,interference=0.0):

        """更新任务状态"""
        if self.start_execute_time is None or self.status=='FINISH':
            self.current_time+=seconds
            return  # 任务未开始或已结束，不需要更新
        
        print(f"\n检查任务 {self.name} 的状态:")
        print(f"  开始时间: {self.start_execute_time}  结束时间: {self.evaluate_finish_time}")
        #计算延迟
        delay = min(self.rescale_time, seconds)#对于推理任务为0
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        self.total_delay += delay
        self.total_delay_with_placement += delay * sum(self.placement)
        seconds -= delay
        #print(f"delay:{delay}")
        #print(f"self current:{self.current_time},seconds:{seconds}")
        if self.current_time + seconds >= self.evaluate_finish_time:#缺训练任务的结束条件
            self.current_time += seconds
            print(f"任务 {self.name} 应该完成了！")
            # 任务完成，从所有分配的GPU中移除
            print(f"正在释放任务 {self.name} 的资源:")
            self.status="FINISH"
            if self.is_inference:
                self.completion_time=self.evaluate_finish_time#让推理任务的completion_time非空
            for gpu_idx in self.allocation:
                gpu = cluster.gpus[gpu_idx]
                if self.is_inference:
                    print(f"  从推理GPU (节点{gpu.node_id}, GPU{gpu.gpu_id}) 释放 {self.requested_gpu}% 资源")
                    gpu.deallocate(self, self.requested_gpu)
                    gpu.protect_start_time = cluster.clock

                else:
                    print(f"  从训练GPU (节点{gpu.node_id}, GPU{gpu.gpu_id}) 释放 100% 资源")
                    gpu.deallocate(self, 100)

            self.allocation = []
            cluster.finished_jobs.add(self)
            cluster.finish_job_set.add(self.name)
            print(f"任务 {self.name} 已完成，完成时间: {self.evaluate_finish_time}")
            print("已完成的任务集合：",cluster.finish_job_set)

            # 打印释放资源后的集群状态
            cluster.print_cluster_status()
            return
        
        while seconds > 0 and self.completion_time is None:
            # 更新训练任务的进度
            # TODO: 在这里添加训练进度的更新逻辑
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
            else:#推理任务
                print("推理任务更新current_time")
                self.current_time += seconds
                print(self.current_time)
                seconds=0#退出循环
        self.current_time+=seconds #把剩下的seconds加上


    def get_job_info(self) -> JobInfo:
        """转换为JobInfo对象，用于策略优化"""
        return JobInfo(
            self.name,
            self.submit_time,
            self.application.name,
            self.target_num_replicas,
            self.requested_gpu,
            self.target_batch_size,
            self.duration,
            self.num_task,
            self.is_inference,
            self.placement,
            self.evaluate_finish_time
        )

class Cluster:
    def __init__(self, num_nodes=16, num_gpus=4, low_util=None, high_util=None):
        self.jobs = []
        self.gpus = []
        self.num_nodes = num_nodes  # 集群的总节点数
        self.num_gpus = num_gpus  # 每个节点的GPU数量
        self.current_time = 0  # 集群的当前时间，用于step函数
        self.clock=0# 模拟的时间进程，用于simulate函数计数
        self.protect_time = 30  # PROTECT状态持续时间
        self.finished_jobs = set()  # 记录已完成的任务
        self.finish_job_set = set()  # 记录已完成的任务名
        self.low_util = low_util  # 利用率下界
        self.high_util = high_util  # 利用率上界
        self.allocations = {}  # 记录所有任务的分配情况，键为任务名，值为GPU下标列表
        self.jobs_submit_time = set()
        self.jobs_finish_time=set()#记录完成时间的集合
        # 初始化策略
        self.train_policy = TrainPolicy()
        self.infer_policy = InferPolicy()
        
        # 初始化训练集群 (前半数个节点，每个节点4个GPU)
        for node_id in range(num_nodes//2):
            for gpu_id in range(num_gpus):
                self.gpus.append(GPU(gpu_id, node_id, False))
        
        # 初始化推理集群 (后8个节点，每个节点4个GPU)
        for node_id in range(num_nodes//2, num_nodes):  # 左闭右开
            for gpu_id in range(num_gpus):
                self.gpus.append(GPU(gpu_id, node_id, True))

    def print_cluster_status(self):
        """打印集群中每个GPU的状态"""
        # print("\n=== GPU集群状态 ===")
        # print("训练集群:")
        # for gpu in self.get_train_gpus():
        #     gpu_idx = self.gpus.index(gpu)
        #     # print(f"GPU {gpu_idx} (节点{gpu.node_id}, GPU{gpu.gpu_id}): 可用空间 {gpu.available_space}%, "
        #     #       f"运行中任务数 {len(gpu.running_jobs)}")
        #     if gpu.running_jobs:
        #         print(f"  GPU{gpu_idx}运行的任务: {[job.name for job in gpu.running_jobs]}")
        
        # print("\n推理集群:")
        # for gpu in self.get_infer_gpus():
        #     gpu_idx = self.gpus.index(gpu)
        #     print(f"GPU {gpu_idx} (节点{gpu.node_id}, GPU{gpu.gpu_id}): 可用空间 {gpu.available_space}%, "
        #           f"状态 {gpu.state}, 运行中任务数 {len(gpu.running_jobs)}")
        #     if gpu.running_jobs:
        #         print(f"  运行的任务: {[job.name for job in gpu.running_jobs]}")

    def all_complete(self) -> bool:
        """检查所有任务是否都已完成"""
        return all(job in self.finished_jobs for job in self.jobs)#所有任务都已完成
        #return all(job.completion_time is not None for job in self.jobs)
        # for job in self.jobs:
        #     if job not in self.finished_jobs:
        #         return False
        # return True

    def load_jobs(self, csv_file: str):
        df = pd.read_csv(csv_file)
        for row in df.itertuples():
            job = Job(
                row.name,
                row.time,
                row.application,
                row.num_replicas,
                row.requested_gpu,
                row.batch_size,
                row.duration,
                row.num_task
            )
            self.jobs.append(job)
            self.jobs_submit_time.add(job.submit_time)

    def get_train_gpus(self) -> List[GPU]:
        return [gpu for gpu in self.gpus if not gpu.is_inference]

    def get_infer_gpus(self) -> List[GPU]:
        return [gpu for gpu in self.gpus if gpu.is_inference]

    def need_update(self):#判断是否有更新的情况，有则需要调用step
        # case1:训练任务固定调度周期
        #is_training_interval = self.clock % args.interval == 0  # 训练任务固定调度周期
        INTERVAL=60
        is_training_interval = self.clock % INTERVAL == 0  # 训练任务固定调度周期

        if is_training_interval:
            LOG.info("case1 schedule interval")
            return True
        # case2:有任务到来
        if self.clock in self.jobs_submit_time:
            return True
        # case3:有任务完成
        if self.clock in self.jobs_finish_time:
            return True
        return False

    def update_gpu_states(self):  # TODO:借出给训练集群时会有BORROW状态
        current_time = self.current_time
        for gpu in self.get_infer_gpus():
            if gpu.state == "PROTECT":
                if current_time - gpu.protect_start_time > self.protect_time:
                    gpu.state = "FREE"

    def apply_allocation(self, job: Job, gpus: List[GPU]):#TODO：把释放资源的逻辑加进来
        """应用分配方案到集群"""
        # 如果任务已经有分配，先释放资源
        if job.allocation:
            print(f"释放任务 {job.name} 的原有资源:")
            for gpu_idx in job.allocation:
                gpu = self.gpus[gpu_idx]
                gpu.deallocate(job, 100)
        if job.is_inference:
            for gpu in gpus:
                gpu.allocate(job, job.requested_gpu)
        else:

            for gpu in gpus:
                gpu.allocate(job, 100)  # 训练任务需要100%的GPU
        
        # 更新任务的allocation和集群的allocations
        job.allocation = [self.gpus.index(gpu) for gpu in gpus]
        print(f"\n任务 {job.name} 分配详情:")
        print(f"  分配的GPU: {job.allocation}")
        
        if job.is_inference:
            print(f"推理任务 {job.name} 已分配")
        else:
            print(f"训练任务 {job.name} 已分配")
            print(f"node_gpu_distribution: {job.node_gpu_distribution}")
            print(f"placement: {job.placement}")

    def cluster_step(self, seconds: int,interval=60):
        #推进仿真进程seconds个时间单位
        #训练集群固定调度周期为interval
        print(f"\n=== 开始新的时间步 ===")
        print(f"当前集群时间: {self.current_time}")
        print(f"当前时钟: {self.clock}")
        
        # 打印当前集群状态
        self.print_cluster_status()

        current_jobs = [job for job in self.jobs if job.submit_time <= self.clock and (
                job.evaluate_finish_time is None or self.current_time < job.evaluate_finish_time)]
        print(f"\n当前待处理任务数量: {len(current_jobs)}")
        for job in self.jobs:
        # for job in current_jobs:
            #job.current_time=self.clock#
            job.job_step(seconds,self)
        
        #job都step完以后进行优化
        self.current_time += seconds

        if current_jobs:
            self.allocations = {
                k: v for k, v in self.allocations.items() if k in current_jobs}#未完成任务的allocation
            # 分离训练任务和推理任务
            train_jobs = [job for job in current_jobs if not job.is_inference]
            infer_jobs = [job for job in current_jobs if job.is_inference]
            for tj in train_jobs:
                print(tj.name, tj.evaluate_finish_time)
            print(f"推理任务数量: {len(infer_jobs)}")
            new_alloc={}
            # 批量优化训练任务
            if train_jobs:
                # 收集训练任务的当前分配情况
                previous_allocation = {}
                for job in train_jobs:
                    if job.allocation:  # 如果任务已有分配
                        previous_allocation[job.name] = [self.gpus[idx] for idx in job.allocation]
                # 获取可用的训练GPU
                available_gpus = self.get_train_gpus()
                # 调用优化函数
                train_allocation = self.train_policy.optimize(
                    [job.get_job_info() for job in train_jobs],
                    previous_allocation,
                    available_gpus
                )
                print("训练优化",train_allocation)
                # 检查每个任务的分配方案是否发生变化
                for job_name, allocated_gpus in train_allocation.items():
                    if job_name in [job.name for job in train_jobs]:
                        job = next(job for job in train_jobs if job.name == job_name)
                        # 获取新的GPU下标列表
                        new_allocation = [self.gpus.index(gpu) for gpu in allocated_gpus]
                        new_alloc.update(new_allocation)#添加训练任务的分配情况
                        # 如果分配方案发生变化，才进行资源分配

            # 批量优化推理任务
            if infer_jobs:
                # 收集推理任务的当前分配情况
                previous_allocation = {}
                for job in infer_jobs:
                    if job.allocation:  # 如果任务已有分配
                        previous_allocation[job.name] = [self.gpus[idx] for idx in job.allocation]
                # 获取可用的推理GPU
                available_gpus = self.get_infer_gpus()
                # 调用优化函数
                infer_allocation, gpu_to_jobs = self.infer_policy.optimize(
                    [job.get_job_info() for job in infer_jobs],
                    previous_allocation,
                    available_gpus
                )

                for job_name, allocated_gpus in infer_allocation.items():
                    # 只处理未完成的任务
                    if job_name in [job.name for job in infer_jobs]:
                        job = next(job for job in infer_jobs if job.name == job_name)
                        # 获取新的GPU下标列表
                        new_allocation = [self.gpus.index(gpu) for gpu in allocated_gpus]
                        new_alloc.update(new_allocation)  # 添加训练任务的分配情况
                        '''
                        # 如果分配方案发生变化，才进行资源分配
                        if set(new_allocation) != set(job.allocation):
                            print(f"任务 {job.name} 的分配方案发生变化:")
                            print(f"  原分配: {job.allocation}  新分配: {new_allocation}")
                            # 如果任务已经有分配，先释放资源
                            if job.allocation:
                                for gpu_idx in job.allocation:
                                    gpu = self.gpus[gpu_idx]
                                    gpu.deallocate(job, job.requested_gpu)
                            # 应用新的分配方案
                            self.apply_allocation(job, allocated_gpus)
                            job.job_allocate()#此处更新evaluate_finish_time
                            job.completion_time=job.evaluate_finish_time
                            self.jobs_finish_time.add(job.evaluate_finish_time)
                            '''

            #TODO：训练任务和推理任务调用完优化方法以后统一进行部署，先根据优化结果的allocations获取placement，再调用job_allocate

            for job in self.jobs:
                if new_alloc.get(job.name) != self.allocations.get(job.name):#此处构建job的placement
                    alloc = new_alloc.get(job.name, [])
                    job.alloc = alloc#意义不明
                    placement = []
                    if job.is_inference:
                        if len(alloc)>0:
                            placement=[1]
                    else:
                        # 计算训练任务的node_gpu_distribution
                        node_gpu_count = {}
                        for gpu in alloc:
                            if gpu.node_id not in node_gpu_count:
                                node_gpu_count[gpu.node_id] = 0
                            node_gpu_count[gpu.node_id] += 1
                        # 按节点ID排序并生成node_gpu_distribution列表，只考虑训练集群的节点（0-7）
                        job.node_gpu_distribution = [node_gpu_count.get(node_id, 0) for node_id in
                                                     range(self.num_nodes // 2)]
                        placement = [count for count in job.node_gpu_distribution if count > 0]
                    #计算完placement布置任务
                    job.job_allocate(placement)#更新Job的开始、预计完成时间、placement
                    self.apply_allocation(job,alloc)
                    # if job.evaluate_finish_time and job.evaluate_finish_time not in self.submit_time:#更新触发update事件的时间点
                    #     self.submit_time[job.evaluate_finish_time] = 1
                    if job.evaluate_finish_time:
                        self.jobs_finish_time.add(job.evaluate_finish_time)
            self.allocations = new_alloc#更新集群的整体分配情况

        # 更新推理GPU的状态
        self.update_gpu_states()
        
        # 输出所有正在运行的训练任务的placement
        running_train_jobs = set()
        for gpu in self.get_train_gpus():
            running_train_jobs.update([job for job in gpu.running_jobs if not job.is_inference and (job.evaluate_finish_time is None or self.current_time < job.evaluate_finish_time)])
        
        if running_train_jobs:
            print("\n当前运行的训练任务:")
            for job in running_train_jobs:
                print(f"任务 {job.name}:")
                print(f"  node_gpu_distribution: {job.node_gpu_distribution}")
                print(f"  placement: {job.placement}")
                print(f"  分配的GPU: {job.allocation}")
                if job.evaluate_finish_time is not None:
                    remaining_time = job.evaluate_finish_time - self.current_time
                    if remaining_time >= 0:  # 只显示未完成的任务
                        print(f"  剩余时间: {remaining_time} 个时间单位")

        print("cluster_step运行结束")


def simulate(args=None):
    # self.jobs.sort(key=lambda x: x.submit_time)
    previous_clock = 0  # 上一次调用step的时间
    # 处理传入的参数args
    INTERVAL = 60  # 从参数读入的训练集群固定调度周期，暂时写成常数
    # 根据policy初始化对应的策略类
    #
    simulator = Cluster()
    simulator.load_jobs("workload/test_train.csv")
    while not simulator.all_complete():
        simulator.clock += 1
        #
        if not simulator.need_update():
            continue
        interval = simulator.clock - previous_clock
        simulator.cluster_step(interval, INTERVAL)  # 推进interval个时间单位的模拟进程


        previous_clock = simulator.clock

    #打印结果日志

if __name__ == "__main__":
    #输入args参数，待添加
    simulate()






