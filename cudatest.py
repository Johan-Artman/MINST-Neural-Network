import os
import ctypes
from ctypes import c_int, byref
from ctypes.util import find_library

# On Windows, ensure the CUDA runtime DLL directory is added to the search path
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin"
    )




def test_cuda_runtime():
    # 1) Försök hitta cudart‐biblioteket
    libname = find_library('cudart')
    if not libname:
        print("Kunde inte hitta CUDA‐runtime‐biblioteket (cudart).")
        print("Kontrollera att CUDA är installerat och att CUDA\\bin ligger i din PATH.")
        return

    # 2) Ladda biblioteket
    try:
        libcudart = ctypes.CDLL(libname)
    except OSError as e:
        print(f"Misslyckades med att ladda {libname}:", e)
        return

    # 3) Hämta funktionen cudaGetDeviceCount
    cudaGetDeviceCount = libcudart.cudaGetDeviceCount
    cudaGetDeviceCount.argtypes = [ctypes.POINTER(c_int)]
    cudaGetDeviceCount.restype  = c_int

    # 4) Anropa den
    count = c_int()
    ret = cudaGetDeviceCount(byref(count))
    if ret != 0:
        print(f"cudaGetDeviceCount returnerade felkod {ret}")
    else:
        print(f"Antal CUDA‐enheter: {count.value}")

if __name__ == "__main__":
    test_cuda_runtime()
