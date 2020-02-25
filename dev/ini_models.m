T_1 = 0.2;
T_2 = 0.1;

params = [T_1,T_2];


T_w = 5.0;
T_pss_1 = 0.61204417;
T_pss_2 = 0.01695186;


params_pss = [T_w,T_pss_1,T_pss_2];

%Tipo T1:
T_c_1 = 2.0
T_b_1 = 10.0
Kpgov_1 = 10.0
Kigov_1 = 2.0
Droop_1 = 0.05
Kimw_1 = 0.01



%Tipo T3:
T_c_3 = 2.0
T_b_3 = 10.0
Kpgov_3 = 10.0
Kigov_3 = 2.0
Droop_3 = 0.05
Kimw_3 = 0.0

params_gov1 = [T_c_1,T_b_1,Kpgov_1,Kigov_1,Droop_1,Kimw_1];
params_gov3 = [T_c_3,T_b_3,Kpgov_3,Kigov_3,Droop_3,Kimw_3];

