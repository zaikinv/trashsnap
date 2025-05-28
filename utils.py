import psutil
import time
import tracemalloc
import subprocess
import streamlit as st

def get_gpu_mem():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        return int(out.decode("utf-8").split("\n")[0])
    except Exception:
        return None

def profile_resources(func):
    def wrapped(*args, **kwargs):
        process = psutil.Process()
        num_cores = psutil.cpu_count(logical=True)
        total_mem = psutil.virtual_memory().total

        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss
        gpu_before = get_gpu_mem()

        tracemalloc.start()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cpu_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss
        gpu_after = get_gpu_mem()

        mem_used_mb = mem_after / 1024**2
        mem_percent = (mem_after / total_mem) * 100

        cpu_percent_total = cpu_after
        cpu_percent_of_system = cpu_after / (num_cores * 100) * 100

        st.info(
            f"""⏱️ {end_time - start_time:.3f}s |
🧠 RAM: {mem_used_mb:.2f} MB ({mem_percent:.2f}%) |
🧮 CPU: {cpu_percent_total:.1f}% (~{cpu_percent_of_system:.1f}% of system capacity)""" +
            (f" | 🖥️ GPU Δ: {gpu_after - gpu_before} MB" if gpu_before and gpu_after else "")
        )

        return result
    return wrapped

