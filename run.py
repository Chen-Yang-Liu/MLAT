import os
import time
import json
import pynvml

if __name__ == '__main__':

    start_time = time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time()))
    print(start_time)

    str = [
        "python ./train5.py",
        "python ./eval.py"
    ]

    k = 0
    while (k < len(str)):
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # 0表示显卡标号
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free = (meminfo.free / 1024 ** 2)
        if free > (3 * 1024):
            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            os.system(str[k])
            print('*************************************************************\n \n')
            k = k + 1
        else:
            print(time.strftime("%m-%d  %H : %M : %S", time.localtime(time.time())))
            time.sleep(10)
        # print(meminfo.total / 1024 ** 2)  # 总的显存大小
        # print(meminfo.used / 1024 ** 2)  # 已用显存大小
        # print(meminfo.free / 1024 ** 2)  # 剩余显存大小











