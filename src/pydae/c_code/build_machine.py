import cffi

ffi = cffi.FFI()

defs = '''
int find_lower_than(double* arr_in, int* arr_out, double x, int n);
double interp1(double* x, double* y, int size, double xi);
double interp2(double* x, double* y, double* z, int nx, int ny, double xi, double yi);
double interp3(double x, double y, double z, double *xVec, int xVec_n, double *yVec, int yVec_n, double *zVec, int zVec_n, double *LUT);
double Flux2id(double t, double theta, double phi_d, double phi_q);
double Flux2iq(double t, double theta, double phi_d, double phi_q);
double Flux2Te(double t, double theta, double phi_d, double phi_q);
'''



ffi.cdef(defs, override=True)

sources = ''
with open('interpolations.c', 'r') as fobj:
    source = fobj.read()
sources += source

with open('machine.c', 'r') as fobj:
    source = fobj.read()
sources += source


ffi.set_source(module_name='machine_cffi',source=sources)
ffi.compile()

# # Example usage
# if __name__ == "__main__":
#     result = multiply(3.5, 2.0)
#     print(f"Multiplication result: {result}")
