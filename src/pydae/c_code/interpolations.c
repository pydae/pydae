#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#define PI 3.14159265358979323846

void main(void){
    ;
}

double interp1(double* x, double* y, int size, double xi) {
    int i, j;
    double yi;

    if (xi <= x[0]) {
        i = 0;
        j = 1;
    } else if (xi >= x[size - 1]) {
        i = size - 2;
        j = size - 1;
    } else {
        for (i = 0; i < size - 1; i++) {
            if (xi >= x[i] && xi <= x[i + 1]) {
                j = i + 1;
                break;
            }
        }
    }
    yi = y[i] + (y[j] - y[i]) * (xi - x[i]) / (x[j] - x[i]);
    return yi;
}


double interp2(double* x, double* y, double* z, int nx, int ny, double xi, double yi) {
    int i, j;
    int i1, i2, j1, j2;
    double x1, x2, y1, y2;
    double f11, f12, f21, f22;
    double dx, dy;

    if (xi <= x[0]) {
        i = 0;
        i1 = i;
        i2 = i + 1;
        x1 = x[i1];
        x2 = x[i2];
    } else if (xi >= x[nx - 1]) {
        i = nx - 2;
        i1 = i;
        i2 = i + 1;
        x1 = x[i1];
        x2 = x[i2];
    } else {
        for (i = 0; i < nx - 1; i++) {
            if (xi >= x[i] && xi < x[i + 1]) {
                i1 = i;
                i2 = i + 1;
                x1 = x[i1];
                x2 = x[i2];
                break;
            }
        }
    }
    
    if (yi <= y[0]) {
        j = 0;
        j1 = j;
        j2 = j + 1;
        y1 = y[j1];
        y2 = y[j2];
    } else if (yi >= y[ny - 1]) {
        j = ny - 2;
        j1 = j;
        j2 = j + 1;
        y1 = y[j1];
        y2 = y[j2];        
    } else {
        for (j = 0; j < ny - 1; j++) {
            if (yi >= y[j] && yi < y[j + 1]) {
                j1 = j;
                j2 = j + 1;
                y1 = y[j1];
                y2 = y[j2];
                break;
            }
        }
    }

    f11 = z[i1 * ny + j1];
    f12 = z[i1 * ny + j2];
    f21 = z[i2 * ny + j1];
    f22 = z[i2 * ny + j2];

    dx = xi - x1;
    dy = yi - y1;

    return (f11 * (x2 - xi) * (y2 - yi) +
            f21 * (xi - x1) * (y2 - yi) +
            f12 * (x2 - xi) * (yi - y1) +
            f22 * (xi - x1) * (yi - y1)) / ((x2 - x1) * (y2 - y1));

}




double interp3(double x, double y, double z, double *xVec, int xVec_n, double *yVec, int yVec_n, double *zVec, int zVec_n, double *LUT){
    int index_x1, index_x0;
    int index_y1, index_y0;
    int index_z1, index_z0;
    int x_aux_n;
    double x_1, x_0, x_d;
    int y_aux_n;
    double y_1, y_0, y_d;
    int z_aux_n;
    double z_1, z_0, z_d;
    int *x_aux = malloc(xVec_n * sizeof(int));
    int *y_aux = malloc(yVec_n * sizeof(int));
    int *z_aux = malloc(zVec_n * sizeof(int));

    if (x >= xVec[xVec_n-1]){
        index_x1 = xVec_n - 1;
        index_x0 = index_x1;
        x_1 = xVec[index_x1];
        x_0 = x_1;
        x_d = 0;
    }else{
        if (x <= xVec[0]){
            index_x0 = 0;
            index_x1 = index_x0;
            x_0 = xVec[index_x0];
            x_1 = x_0;
            x_d = 0;
        }else{
            x_aux_n = find_lower_than(xVec, x_aux, x, xVec_n);
            index_x0 = x_aux[x_aux_n - 1];
            index_x1 = x_aux[x_aux_n - 1] + 1;
            x_0 = xVec[index_x0];
            x_1 = xVec[index_x1];
            x_d = (x - x_0)/(x_1 - x_0);
        }
    }

    

    if (y >= yVec[yVec_n-1]){
        index_y1 = yVec_n - 1;
        index_y0 = index_y1;
        y_1 = yVec[index_y1];
        y_0 = y_1;
        y_d = 0;
    }else{
        if (y <= yVec[0]){
            index_y0 = 0;
            index_y1 = index_y0;
            y_0 = yVec[index_y0];
            y_1 = y_0;
            y_d = 0;
        }else{
            y_aux_n = find_lower_than(yVec, y_aux, y, yVec_n);
            index_y0 = y_aux[y_aux_n - 1];
            index_y1 = y_aux[y_aux_n - 1] + 1;
            y_0 = yVec[index_y0];
            y_1 = yVec[index_y1];
            y_d = (y - y_0)/(y_1 - y_0);
        }
    }

    if (z >= zVec[zVec_n-1]){
        index_z1 = zVec_n - 1;
        index_z0 = index_z1;
        z_1 = zVec[index_z1];
        z_0 = z_1;
        z_d = 0;
    }else{
        if (z <= zVec[0]){
            index_z0 = 0;
            index_z1 = index_z0;
            z_0 = zVec[index_z0];
            z_1 = z_0;
            z_d = 0;
        }else{
            z_aux_n = find_lower_than(zVec, z_aux, z, zVec_n);
            index_z0 = z_aux[z_aux_n - 1];
            index_z1 = z_aux[z_aux_n - 1] + 1;
            z_0 = zVec[index_z0];
            z_1 = zVec[index_z1];
            z_d = (z - z_0)/(z_1 - z_0);
        }
    }

    double C_000 = LUT[zVec_n*yVec_n*index_x0 + zVec_n*index_y0 + index_z0];
    double C_001 = LUT[zVec_n*yVec_n*index_x0 + zVec_n*index_y0 + index_z1];
    double C_010 = LUT[zVec_n*yVec_n*index_x0 + zVec_n*index_y1 + index_z0];
    double C_011 = LUT[zVec_n*yVec_n*index_x0 + zVec_n*index_y1 + index_z1];
    double C_100 = LUT[zVec_n*yVec_n*index_x1 + zVec_n*index_y0 + index_z0];
    double C_101 = LUT[zVec_n*yVec_n*index_x1 + zVec_n*index_y0 + index_z1];
    double C_110 = LUT[zVec_n*yVec_n*index_x1 + zVec_n*index_y1 + index_z0];
    double C_111 = LUT[zVec_n*yVec_n*index_x1 + zVec_n*index_y1 + index_z1];

    double C_00 = C_000*(1-x_d) + C_100*x_d;
    double C_01 = C_001*(1-x_d) + C_101*x_d;
    double C_10 = C_010*(1-x_d) + C_110*x_d;
    double C_11 = C_011*(1-x_d) + C_111*x_d;

    double C_0 = C_00*(1-y_d) + C_10*y_d;
    double C_1 = C_01*(1-y_d) + C_11*y_d;

    double C = C_0*(1-z_d) + C_1*z_d;
    
    free(x_aux);
    free(y_aux);
    free(z_aux);

    return C;
}


int find_lower_than(double* arr_in, int* arr_out, double x, int n){

    int pointer = 0;
    for (int i = 0; i < n; i++) {
        if (arr_in[i] <= x){
            arr_out[pointer] = i;
            pointer++;
        }
    }
    return pointer;
}


