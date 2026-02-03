# Importing required libraries
import numpy as np
from scipy.optimize import fsolve, minimize, NonlinearConstraint



class grid:
    def __init__(self, nodes, lines, pros):
        self.nodes = self.add_nodes(nodes)                                      
        self.lines = self.add_lines(lines, self.nodes)  
        self.pros = self.add_pros(pros, self.nodes)  
        self.pf = pf(self)
        self.ofo = ofo(self)
        self.opf = opf(self)
                
    def add_nodes(self, nodes):
        nodes_list = list()
        for item in nodes:
            nodes_list.append(node(item['id'], item['slack']))
        return nodes_list
        
    def add_lines(self, lines, nodes):
        lines_list = list()
        for item in lines:
            lines_list.append(line(item['id'], item['From'], item['To'], item['R'], item['X'], nodes))
        return lines_list
        
    def add_pros(self, pros, nodes):
        pros_list = list()
        for item in pros:
            pros_list.append(prosumer(item['id'], item['Node'], item['P'], item['Q'], nodes))
        return pros_list


class opf:
    def __init__(self, net):
        self.net = net
                
    def set_params(self, params):
        self.inputs, self.outputs = [], []
        for item in params:
            if item['type'] == 'input':
                name = 'ofo-' + item['type'] + '-' + item['signal'] + '-' + item['element'] + '-' + str(item['node'])
                self.inputs.append((name, item['bounds']))
                self.net.pros.append(prosumer(name, item['node'], 0, 0, self.net.nodes))
                self.net.pros[-1].bnds = item['bounds']
            if item['type'] == 'output':
                name = 'ofo-' + item['type'] + '-' + item['signal'] + '-' + item['element'] + '-' + str(item[item['element']])
                self.outputs.append((name, item['bounds']))
        self.n_in = len(self.inputs)
        self.n_out = len(self.outputs)
    
    def set_inputs(self, inputs):
        in_base = self.get_inputs()
        for k, v in zip(in_base.keys(), inputs):
            in_base[k] = v
        for in_item in in_base:
            _, _, mag, elem, ref = in_item.split('-') 
            pro = next((item for item in self.net.pros if item.ref == in_item), None)
            setattr(pro, mag, in_base[in_item])
        
    def get_inputs(self):
        in_base = {item[0]: 0 for item in self.inputs}
        self.net.pf.solve_pf()
        for in_item in in_base:
            _, _, mag, elem, ref = in_item.split('-') 
            pro = next((item for item in self.net.pros if item.ref == in_item), None)
            in_base[in_item] = getattr(pro, mag)
        return in_base
        
    
    def get_outputs(self):
        self.output_data = []
        for item in [output[0] for output in self.outputs]:
            _, _, mag, elem, ref = item.split('-') 
            if elem == 'node':
                self.output_data.append(np.abs(getattr(self.net.nodes[int(ref)], mag)))
            if elem == 'line':
                self.output_data.append(np.abs(getattr(self.net.lines[int(ref)], mag)))
        return self.output_data
    
    def obj_function(self, x):
        self.set_inputs(x)
        self.net.pf.solve_pf()
        y = self.get_outputs()
        return self.f_u @ x + self.f_y @ y 
    
    def y_constraints(self, x):
        self.set_inputs(x)
        self.net.pf.solve_pf()
        y = self.get_outputs()
        return self.C @ y - self.d
    
    def solve_opf(self, obj):
        self.C = np.block([[ np.eye(self.n_out)],
                           [-np.eye(self.n_out)]])
        self.d = np.array([item[1][1] for item in self.outputs] + [-item[1][0] for item in self.outputs])
        self.f_u, self.f_y = obj          
        y_constr = NonlinearConstraint(self.y_constraints, -np.inf, 0)
        sol = minimize(self.obj_function, 
                       np.array(list(self.get_inputs().values())), 
                       bounds=[(item[1][0], item[1][1]) for item in self.inputs],
                       constraints=(y_constr,))
        print(f'OPF success: {sol.success}, obj: {sol.fun}\n')
        return sol
        
        
        
        
        

class ofo:
    def __init__(self, net):
        self.net = net
        
    def set_params(self, params):
        self.inputs, self.outputs = [], []
        for item in params:
            if item['type'] == 'input':
                name = 'ofo-' + item['type'] + '-' + item['signal'] + '-' + item['element'] + '-' + str(item['node'])
                self.inputs.append((name, item['bounds']))
                self.net.pros.append(prosumer(name, item['node'], 0, 0, self.net.nodes))
                self.net.pros[-1].bnds = item['bounds']
            if item['type'] == 'output':
                name = 'ofo-' + item['type'] + '-' + item['signal'] + '-' + item['element'] + '-' + str(item[item['element']])
                self.outputs.append((name, item['bounds']))
        self.n_in = len(self.inputs)
        self.n_out = len(self.outputs)
    
    def reset(self):
        u = self.get_inputs()
        for k, v in zip(u.keys(), np.zeros(len(u))):
            u[k] = v
        self.set_inputs(u)
        self.net.pf.solve_pf()
        y = self.get_outputs()
    
    def set_inputs(self, inputs):
        for in_item in inputs:
            if len(in_item) == 2:
                _, _, mag, elem, ref = in_item[0].split('-') 
                pro = next((item for item in self.net.pros if item.ref == in_item[0]), None)
                setattr(pro, mag, inputs[in_item])
            else:
                _, _, mag, elem, ref = in_item.split('-') 
                pro = next((item for item in self.net.pros if item.ref == in_item), None)
                setattr(pro, mag, inputs[in_item])
        
    def get_inputs(self):
        in_base = {item[0]: 0 for item in self.inputs}
        self.net.pf.solve_pf()
        for in_item in in_base:
            _, _, mag, elem, ref = in_item.split('-') 
            pro = next((item for item in self.net.pros if item.ref == in_item), None)
            in_base[in_item] = getattr(pro, mag)
        return in_base
        
    
    def get_outputs(self):
        self.output_data = []
        for item in [output[0] for output in self.outputs]:
            _, _, mag, elem, ref = item.split('-') 
            if elem == 'node':
                self.output_data.append(np.abs(getattr(self.net.nodes[int(ref)], mag)))
            if elem == 'line':
                self.output_data.append(np.abs(getattr(self.net.lines[int(ref)], mag)))
        return self.output_data
            
    
    def compute_H(self):
        self.net.pf.solve_pf()
        self.H = np.zeros((self.n_out, self.n_in))
        in_base = self.get_inputs()
        for idx in range(self.n_in):
            self.H[:, idx] = self.get_outputs()
        for idx, key in enumerate(in_base):
            in_base[key] += 1
            self.set_inputs(in_base)
            self.net.pf.solve_pf()
            self.H[:, idx] = - self.H[:, idx] + self.get_outputs()
            in_base[key] -= 1        
        self.set_inputs(in_base)
        self.net.pf.solve_pf()
          
    def solve_ofo(self, obj, alpha = 0.01, rho = 1, max_iter = 10):
        self.reset()
        self.compute_H()
        self.C = np.block([[ np.eye(self.n_out)],
                           [-np.eye(self.n_out)]])
        self.d = np.array([item[1][1] for item in self.outputs] + [-item[1][0] for item in self.outputs])
        self.mu = np.zeros(self.C.shape[0])
        self.s = np.zeros(self.C.shape[0])
        self.f_u, self.f_y = obj          
        self.in_bnds = [(item[1][0], item[1][1]) for item in self.inputs] 
                    
        # Data aquisition
        hist = {'u_h': [], 'y_h': [], 'loss_h': [], 'of_h': [], 'mu_h': [], 'H_h': []}
        actual_inputs = np.array(list(self.get_inputs().values()))
        actual_outputs = np.array(self.get_outputs())
        
        for it in range(max_iter):
            # Verbose
            print(f'Iteration {it}')
            
            # Updating slack variables
            self.s = np.clip((-(1/rho)*self.mu - self.C @ actual_outputs + self.d), a_min = 0, a_max=None)
            
            # Updating primal variables
            delta_u = self.f_u + self.H.T @ self.f_y + \
                self.H.T @ self.C.T @ (self.mu + rho*(self.C @ actual_outputs + self.s - self.d))
            actual_inputs = actual_inputs - alpha*delta_u
            actual_inputs = np.clip(actual_inputs, a_min = [bnds[0] for bnds in self.in_bnds], a_max = [bnds[1] for bnds in self.in_bnds])
            actual_inputs_dict = self.get_inputs()
            for k, v in zip(actual_inputs_dict.keys(), actual_inputs):
                actual_inputs_dict[k] = v
            self.set_inputs(actual_inputs_dict)       
                                    
            # Solving power flow and receiving outputs
            self.net.pf.solve_pf()
            actual_outputs = np.array(self.get_outputs())
            
            # Updating sensitivity martix
            self.compute_H()
                        
            # Updating dual variables
            self.mu += rho*(self.C @ actual_outputs - self.d + self.s)
            
            # Saving data    
            hist['u_h'].append(actual_inputs.copy())
            hist['y_h'].append(actual_outputs.copy())
            hist['mu_h'].append(self.mu.copy())
            hist['H_h'].append(self.H.copy())
          
        print(f'OFO final value: {self.f_u @ actual_inputs + self.f_y @ actual_outputs}\n')
        return hist

class pf:
    def __init__(self, net):
        self.net = net

    def solve_pf(self, verbose = False):
        x0 = [1,0]*len(self.net.nodes)
        sol, infodict, ier, mesg = fsolve(self.test_x, x0, full_output = True)
        if verbose:
            print(mesg)
        index = 0
        for node in self.net.nodes:
            node.U = complex(sol[index], sol[index+1])
            index += 2
        for line in self.net.lines:
            line.compute_I()
        for line in self.net.lines:
            line.compute_PQ()
        return sol, infodict, ier, mesg
    
    def compute_res(self):
        residual = []
        for node in self.net.nodes:
            residual.append(node.check())
        residual_rx = []
        for item in residual:
            residual_rx.append(np.real(item))
            residual_rx.append(np.imag(item))
        return residual_rx
    
    def test_x(self, x):
        self.assign_x(x)
        self.compute_I()
        res = self.compute_res()
        return res    
    
    def assign_x(self, x):
        index = 0
        for node in self.net.nodes:
            node.U = complex(x[index], x[index + 1])
            index += 2   
            
    def compute_I(self):
        for line in self.net.lines:
            line.I = (line.nodes[0].U - line.nodes[1].U)/line.Z
        for node in self.net.nodes: # +: inyeccion, -: demanda
            node.I = np.conj(np.sum([complex(p.P, p.Q) for p in node.pros])/node.U)
            
class node:
    def __init__(self, ref, slack):
        self.ref = ref   
        self.slack = slack        
        self.lines = list()
        self.U = complex(1, 0)
        self.pros = []
    
    def check(self):
        if self.slack:
            residual = self.U - complex(1, 0)
        else:
            I_agregada = 0
            I_agregada += self.I
            for line in self.lines:
                if line.nodes[0] == self:
                    I_agregada -= line.I
                else:
                    I_agregada += line.I
            residual = I_agregada
        return residual
        
class line:
    def __init__(self, ref, From, To, R, X, nodes_list):
        self.ref = ref     
        self.Z = complex(R, X)
        self.G, self.B = np.real(1/self.Z), -np.imag(1/self.Z)
        self.Y = 1/self.Z
        self.nodes = [next((item for item in nodes_list if item.ref == From), None), 
                      next((item for item in nodes_list if item.ref == To), None)]   
        self.nodes[0].lines.append(self)
        self.nodes[1].lines.append(self)
        
    def compute_I(self):
        self.I = (self.nodes[0].U - self.nodes[1].U)/self.Z
        
    def compute_PQ(self):
        self.P = np.real(self.nodes[0].U*np.conj(self.I))
        self.Q = np.imag(self.nodes[0].U*np.conj(self.I))
            
  
class prosumer:
    def __init__(self, ref, node_id, P, Q, nodes_list):
        self.ref = ref
        self.P = P
        self.Q = Q        
        self.node = next((item for item in nodes_list if item.ref == node_id), None)
        self.node.pros.append(self)
        
        
class DTR_line:     
    def __init__(self):
        self.D = 27.7e-3
        self.d = 3.08e-3
        self.sigma  = 5.670373e-8
        self.epsilon = 0.8
        self.alpha = 0.8
        self.Tc_max = 95 
        self.Rac = 1.04*0.000071873143*(1+0.000937474*(self.Tc_max-20));
        self.h = 1
        self.g = 9.81
    
    def compute_I(self, Tamb, v_w, ang_w, G):
        self.Tamb = Tamb
        self.v_w = v_w
        self.ang_w = ang_w
        self.G = G
        
        self.compute_Pr()
        self.compute_Pc()
        self.compute_Ps()
        
        self.I = np.sqrt( (self.Pr + self.Pc - self.Ps)/self.Rac )        
        return self.I
        
    def compute_Pr(self):
        self.Pr = np.pi*self.D*self.sigma*self.epsilon*((self.Tc_max + 273.15)**4 - (self.Tamb + 273.15)**4)
        return self.Pr
        
    def compute_Pc(self):        
        Tf = 0.5*(self.Tamb + self.Tc_max)
        lmb = 2.368e-2 + 7.23e-5*Tf - 2.763e-8*Tf**2
        cf = 1.9327e-10*Tf**4 - 7.9999e-7*Tf**3 + 1.1407e-3*Tf**2 - 0.4489*Tf + 1057.5
        Rs = self.d / (2*(self.D - self.d))
        
        mu_f = (17.239 + 4.635e-2*Tf - 2.03e-5*Tf**2) * 1e-6
        dens = (1.293 - 1.525e-4*self.h + 6.379e-9*self.h**2) / (1 + 0.00367*Tf)
        uf = mu_f / dens  
        Re = self.v_w * self.D / uf

        B, n = (0.178, 0.633) if (Re > 2650 and Rs <= 0.05) else ((0.048, 0.8) if Re > 2650 else (0.641, 0.471))
        Nu90 = B * Re**n

        angle = self.ang_w if self.ang_w <= 180 else self.ang_w - 180
        seno = np.abs(np.sin(np.radians(angle)))
        delta = 0.42 + (0.68 * seno**1.08 if angle <= 24 else 0.58 * seno**0.9)
        Nu_forz = Nu90 * delta
        
        Gr = self.D**3 * (self.Tc_max - self.Tamb) * self.g / ((Tf + 273.15) * uf**2)
        Pr = cf * mu_f / lmb
        prod = Gr * Pr

        if 0.1 < prod < 1e2:
            A, m = 1.02, 0.148
        elif 1e2 <= prod < 1e4:
            A, m = 0.85, 0.188
        elif 1e4 <= prod < 1e7:
            A, m = 0.48, 0.25
        elif 1e7 <= prod < 1e12:
            A, m = 0.125, 0.333
        else:
            A, m = 0, 0
        Nu_nat = A * prod**m

        Nu = np.max([Nu_nat, Nu_forz])

        self.Pc = np.pi * lmb * (self.Tc_max - self.Tamb) * Nu
        return self.Pc
    
    def compute_Ps(self):
        self.Ps = self.alpha*self.G*self.D
        return self.Ps
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    