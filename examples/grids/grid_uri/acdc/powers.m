syms v_a_r v_b_r v_c_r real
syms v_a_i v_b_i v_c_i real
syms i_a_r i_b_r i_c_r real
syms i_a_i i_b_i i_c_i real
syms a_r a_i real

alpha = exp(2.0/3*pi*j);
alpha = a_r + j*a_i
%a_r,a_i = sym.symbols('a_r,a_i',real=True)
%alpha = a_r + sym.I*a_i

v_abc =  [[v_a_r+j*v_a_i];[v_b_r+j*v_b_i];[v_c_r+j*v_c_i]]; 
i_abc =  [[i_a_r+j*i_a_i];[i_b_r+j*i_b_i];[i_c_r+j*i_c_i]]; 


A_0a =  [[1, 1, 1],
         [1, alpha^2, alpha],
         [1, alpha, alpha^2]] ;

A_a0 = 1./3* [[1, 1, 1],
             [1, alpha, alpha^2],
             [1, alpha^2, alpha]];

i_opn = A_a0*i_abc;
v_opn = A_a0*v_abc;

s_abc = v_abc'*i_abc;
s_opn = 3.0*v_opn'*i_opn;

simplify(s_opn-s_abc)

s_opn = simplify(3*v_opn(2)'*i_opn(2))