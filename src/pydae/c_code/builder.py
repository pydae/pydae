import cffi

ffi = cffi.FFI()

with open('pwm.h', 'r') as fobj:
    defs = fobj.read()

with open('pwm.c', 'r') as fobj:
    source = fobj.read()


ffi.cdef(defs, override=True)
ffi.set_source(module_name='pwm_cffi',source=source)
ffi.compile()

# # Example usage
# if __name__ == "__main__":
#     result = multiply(3.5, 2.0)
#     print(f"Multiplication result: {result}")
