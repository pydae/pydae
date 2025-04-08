int find_lower_than(double* arr_in, int* arr_out, double x, int n);
double interp1(double* x, double* y, int size, double xi);
double interp2(double* x, double* y, double* z, int nx, int ny, double xi, double yi);
double interp3(double x, double y, double z, double *xVec, int xVec_n, double *yVec, int yVec_n, double *zVec, int zVec_n, double *LUT);
