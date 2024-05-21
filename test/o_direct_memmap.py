import numpy
import os
import mmap
import ctypes

from numpy.lib.format import dtype_to_descr, _check_version, _write_array_header, read_magic, _read_array_header, _MAX_HEADER_SIZE

# 定义文件系统块大小，假设为4096字节
BLOCK_SIZE = 4096

def open_memmap(filename, mode='r+', dtype=None, shape=None,
                fortran_order=False, version=None, *,
                max_header_size=_MAX_HEADER_SIZE):

    if isinstance(filename, (str, bytes, os.PathLike)):
        filename = os.fspath(filename)
    else:
        raise ValueError("Filename must be a string or a path-like object. "
                         "Memmap cannot use existing file handles.")

    if 'w' in mode:
        # We are creating the file, not reading it.
        # Check if we ought to create the file.
        _check_version(version)
        # Ensure that the given dtype is an authentic dtype object rather
        # than just something that can be interpreted as a dtype object.
        dtype = numpy.dtype(dtype)
        if dtype.hasobject:
            msg = "Array can't be memory-mapped: Python objects in dtype."
            raise ValueError(msg)
        d = dict(
            descr=dtype_to_descr(dtype),
            fortran_order=fortran_order,
            shape=shape,
        )
        # If we got here, then it should be safe to create the file.
        with open(filename, mode+'b') as fp:
            _write_array_header(fp, d, version)
            offset = fp.tell()
    else:
        # Read the header of the file first.
        with open(filename, 'rb') as fp:
            version = read_magic(fp)
            _check_version(version)

            shape, fortran_order, dtype = _read_array_header(
                    fp, version, max_header_size=max_header_size)
            if dtype.hasobject:
                msg = "Array can't be memory-mapped: Python objects in dtype."
                raise ValueError(msg)
            offset = fp.tell()

    if fortran_order:
        order = 'F'
    else:
        order = 'C'

    # We need to change a write-only mode to a read-write mode since we've
    # already written data to the file.
    if mode == 'w+':
        mode = 'r+'

    # 使用 os.open 打开文件，并添加 os.O_DIRECT 标志
    fd = os.open(filename, os.O_RDWR | os.O_DIRECT)
    length = numpy.prod(shape) * dtype.itemsize
    # 确保读写长度是文件系统块大小的倍数
    aligned_length = ((length + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    
    # 使用 bytearray 初始化缓冲区
    buf = bytearray(aligned_length + BLOCK_SIZE)
    address = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    offset_address = (address + BLOCK_SIZE - 1) & ~(BLOCK_SIZE - 1)
    
    if offset_address % BLOCK_SIZE != 0:
        raise ValueError("Buffer address is not aligned to block size")

    # 创建内存映射对象
    marray = numpy.memmap(filename, dtype=dtype, shape=shape, order=order,
                          mode=mode, offset=offset)

    # 关闭文件描述符
    os.close(fd)

    return marray