import ctypes
import mmap


def alloc_shared_matrix(rows, cols):
    total = rows * cols * ctypes.sizeof(ctypes.c_float)
    mm = mmap.mmap(-1, total)
    base = ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(mm)))

    RowArray = ctypes.POINTER(ctypes.c_float) * rows
    mat = RowArray()

    for i in range(rows):
        addr = base.value + i * cols * ctypes.sizeof(ctypes.c_float)
        mat[i] = ctypes.cast(addr, ctypes.POINTER(ctypes.c_float))

    return mat, base, mm


def list_to_shared(mat):
    assert len(mat) > 0, "list_to_shared received empty matrix"
    assert len(mat[0]) > 0, "list_to_shared received zero-width matrix"
    rows, cols = len(mat), len(mat[0])
    m, base, mm = alloc_shared_matrix(rows, cols)
    for i in range(rows):
        for j in range(cols):
            m[i][j] = float(mat[i][j])
    return m, base, mm


def shared_to_list(m, rows, cols):
    return [[float(m[i][j]) for j in range(cols)] for i in range(rows)]
