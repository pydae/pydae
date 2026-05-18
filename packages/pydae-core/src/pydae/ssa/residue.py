import numpy as np

def residue(model, input_name, output_name, mode_idx):
    """
    Calculates the residue and required phase compensation for a Power System Stabilizer (PSS).
    
    This function uses the state-space matrices of a PyDAE model to evaluate the 
    transfer function residue between a specific actuator (input) and sensor (output) 
    for a given electromechanical mode. It then computes the required phase lead/lag 
    to achieve pure damping (180 degrees closed-loop phase).

    Args:
        model (pydae.models): The initialized PyDAE model containing state-space matrices (A, B, C).
        input_name (str): The name of the input variable (e.g., 'v_ref_LT').
        output_name (str): The name of the output measurement variable (e.g., 'omega_LT' or 'p_e_LT').
        mode_idx (int): The index of the targeted eigenvalue/mode in the model's eigenvalue array.
        lambda_i (complex): The complex eigenvalue of the targeted mode.

    Returns:
        tuple: A tuple containing:
            - residue_angle_deg (float): The angle of the residue in degrees.
            - phi_lead_deg (float): The required PSS phase compensation in degrees.
    """
    
    lambda_i = model.eigenvalues[mode_idx]

    # 1. Plant input formulation
    if input_name in model.u_run_names:
        idx_u = model.u_run_names.index(input_name) 
    else:
        raise ValueError(f"Error: {input_name} is not a valid input parameter.")
        
    B_pss = model.B[:, idx_u].reshape(-1, 1)

    # 2. Plant output formulation
    if output_name in model.z_list:
        idx_z = model.z_list.index(output_name)      
        C_meas = model.C[idx_z, :].reshape(1, -1)
    elif output_name in model.x_list:
        C_meas = np.zeros((1, model.N_x))
        C_meas[0, model.x_list.index(output_name)] = 1
    else:
        raise ValueError(f"Error: {output_name} is neither an algebraic output nor a dynamic state.")

    # 3. Extract the right (v_i) and left (w_i) eigenvectors for the selected mode
    v_i = model.right_eigenvectors[:, mode_idx].reshape(-1, 1)
    w_i = model.left_eigenvectors[mode_idx, :].reshape(1, -1)

    # 4. Calculate the Residue (R_i = C * v_i * w_i * B)
    R_i = (C_meas @ v_i) * (w_i @ B_pss)
    residue_angle_deg = np.angle(R_i[0, 0], deg=True)

    # 5. Calculate the required Phase Compensation (For negative feedback loop)
    # We want the total closed-loop phase to point to 180º (pure damping)
    phi_lead_deg = 180.0 - residue_angle_deg

    # 6. Normalize the angle to keep it in the [-180, 180] range
    phi_lead_deg = (phi_lead_deg + 180) % 360 - 180

    # 7. Print the report
    print('\n')
    print("-" * 50)
    print(f"Residue for {input_name} -> {output_name}:")
    print("-" * 50)
    print(f"Analyzed mode                   : {lambda_i:.4f}")
    print(f"Natural frequency (rad/s)       : {np.imag(lambda_i):.4f}")
    print(f"Residue Angle (R_i)             : {residue_angle_deg:.2f}º")
    print(f"Required PSS Compensation       : {phi_lead_deg:.2f}º")
    print("-" * 50)
    
    return R_i



# # --- Usage Example ---
# # Assuming `model`, `mode_idx`, and `lambda_i` have already been computed:
# input_name = 'v_ref_LT'
# output_name = 'omega_LT'

# # Call the function and optionally store the results
# r_angle, pss_comp = residue(model, input_name, output_name, mode_idx, lambda_i)