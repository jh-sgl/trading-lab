import os
import time
import psutil
import numpy as np
import subprocess

def get_average_cpu_load_over_time(interval=1, duration=5):
    # 각 CPU 코어에 대한 부하를 저장할 리스트를 초기화합니다.
    cpu_loads = {i: [] for i in range(psutil.cpu_count())}
    
    # 지정된 시간(duration) 동안 CPU 사용률을 수집합니다.
    start_time = time.time()
    while time.time() - start_time < duration:
        # 각 코어에 대한 현재 CPU 사용률을 얻습니다.
        cpu_percents = psutil.cpu_percent(percpu=True)
        for i, percent in enumerate(cpu_percents):
            cpu_loads[i].append(percent)
        
        # 지정된 간격(interval)만큼 대기합니다.
        time.sleep(interval)
    
    # 각 코어에 대한 평균 부하를 계산합니다.
    average_loads = {i: np.mean(loads) for i, loads in cpu_loads.items()}
    return average_loads

def get_least_busy_cores(average_loads, num_cores=8):
    # 평균 부하가 낮은 순으로 코어를 정렬합니다.
    sorted_cores = sorted(average_loads, key=average_loads.get)
    # 가장 부하가 낮은 num_cores개의 코어를 반환합니다.
    return sorted_cores[:num_cores]

def set_affinity(cores):
    pid = os.getpid()
    os.sched_setaffinity(pid, cores)

def get_free_gpu():
    smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
    )
    gpu_info = smi_query_result.decode("utf-8").split("\n")

    gpu_total = list(filter(lambda info: "Total" in info, gpu_info))
    gpu_used = list(filter(lambda info: "Used" in info, gpu_info))
    
    gpu_total = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_total]
    gpu_used = [int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_used]
    
    gpu_free = np.array(gpu_total) - np.array(gpu_used)
    
    free_id = gpu_free.argmax()
    
    if free_id == 4:
        return 3
    
    if free_id == 3:
        return None
    
    return gpu_free.argmax()