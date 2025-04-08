#include <stdio.h>
#include <math.h>

double PWM(double t, double eta, double freq, double T_l){
    double carrier;
    double s;
    double pi = 3.1415926; 
    double T_s = 0.5e-6;
    double s_avg;
    int N_s;
    double t_s;

    
    N_s = (int)(T_l/T_s);
    s_avg = 0.0;

    for (int k = 1; k<= N_s; k++) {

        t_s = t - T_l/2 + T_s*k;
        carrier = 2.0*(asin(cos(2*pi*freq*t_s))/pi);

        s=-1.0;
        if(eta>carrier){
            s = 1.0;
        }

        s_avg += s*T_s/T_l;

    }
  
    return s_avg;
}

