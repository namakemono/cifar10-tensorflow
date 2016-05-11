from __future__ import division
from __future__ import print_function
from __future__ import with_statement
import pynvml as nvml


nice_ratio = 1/3

# Usage:
#   frac = gentleman.request_mem(4*1024) # politely req ~4G
#   gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=frac)
#   sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts))
def request_mem(mem_mb, i_am_nice=True):
    # titanx' mem:        12,881,559,552 bytes
    # 12*1024*1024*1024 = 12,884,901,888
    mem = mem_mb * 1024 * 1024
    nvml.nvmlInit()
    # n = nvml.nvmlDeviceGetCount()
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info   = nvml.nvmlDeviceGetMemoryInfo(handle)
        cap = info.total * nice_ratio
        # req = cap if mem > cap and i_am_nice else mem
        req = mem
        if req > cap and i_am_nice:
            raise MemoryError('You are supposed to be polite..')
        if req > info.free:
            raise MemoryError('Cannot fullfil the gpumem request')
        return req / info.free
    finally:
        nvml.nvmlShutdown()


if __name__ == '__main__':
    print('nice:', request_mem(1024*3.9))
    print('rude:', request_mem(1024*5, False))
    print('rude:', request_mem(1024*12, False))

