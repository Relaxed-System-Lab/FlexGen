import os
import mmap
import numpy as np

# 文件路径
file_path = 'w4-0'

# 打开文件，使用 O_DIRECT 标志
fd = os.open(file_path, os.O_RDWR | os.O_DIRECT)

# 获取文件大小
file_size = os.path.getsize(file_path)

# 创建内存映射对象
# mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_WRITE)
mm = mmap.mmap(fd, file_size, access=mmap.ACCESS_READ)



# 将内存映射对象转换为 NumPy 数组
# 假设文件中存储的是 float64 类型的数据
array = np.frombuffer(mm, dtype=np.float64)

# 关闭内存映射对象和文件描述符
mm.close()
os.close(fd)

# 现在你可以像操作普通 NumPy 数组一样操作 array
print(array)