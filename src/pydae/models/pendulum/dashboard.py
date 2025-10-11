import panel as pn
import datetime
import param
import asyncio
import re
from pydae.svg_tools import svg
import requests
import json
import time
import matplotlib.pyplot as plt
import numpy as np
pn.extension('katex','matplotlib')

t_0 = time.time()
plot_pane = pn.pane.Matplotlib(sizing_mode='stretch_width')

# A function to generate a new plot

# with open('cuerva_emec_db.svg', 'r') as fobj:
#     svg_file_content = fobj.read()

# Generate the new SVG
#s = svg('cuerva_emec_db.svg')
# s.set_tspan('U_SS1', f"{meas_dict['V_SS1']:5.3f} pu")
# s.set_tspan('U_SS2', f"{meas_dict['V_SS2']:5.3f} pu")
# s.set_tspan('U_POI', f"{slider_val:5.3f} pu")
# s.set_title('GRID_title', f"V = {meas_dict['V_GRID']:5.3f} pu\nP = {meas_dict['p_line_POIHV_GRID'] / 1e6:5.3f} MW\nQ = {meas_dict['q_line_POIHV_GRID'] / 1e6:5.3f} Mvar")
# s.set_tspan('P_POI', f"{meas_dict['p_line_POI_POIHV'] / 1e6:5.1f} MW")
# s.set_tspan('Q_POI', f"{meas_dict['q_line_POI_POIHV'] / 1e6:5.1f} Mvar")

# html_code = """
# <div>
#   <svg width="200" height="200" xmlns="http://www.w3.org/2000/svg">
#     <circle id="interactive_circle" cx="100" cy="100" r="50" fill="skyblue">
#       <title>This is a circle\nHola</title>
#     </circle>
#   <text id="time_label" x="50" y="50" font-family="Arial" font-size="24" text-anchor="middle" fill="#333">
#     Time: 00:00:00
#   </text>
#   </svg>
# </div>
# """

N_t = 300
charts_data = {'Times':np.zeros(N_t),
               'theta':np.zeros(N_t),
               'f_x':np.zeros(N_t),
               'k':0, 'N_t':N_t, 't_0':t_0
               }

# html_code = f"""
# <div>
#   {s.tostring()}
# </div>
# """


#svg_pane = pn.pane.HTML(html_code, width=200, height=200)
# panel.servable()


# Create other widgets
button1 = pn.widgets.Button(name='Set to Red', button_type='success')
button2 = pn.widgets.Button(name='Set to Green', button_type='primary')
slider1 = pn.widgets.IntSlider(name='Circle Radius', start=20, end=100, value=80)
color_picker = pn.widgets.ColorPicker(name='Circle Color', value='#FF0000')

 
sld_f_x_label = pn.pane.LaTeX(r'$\mathsf {f_{x}}$ (Nm):', width=80)
sld_f_x = pn.widgets.FloatSlider(start=-200.0, end=200.0, value=0.0, step=0.01)

# Crear pestañas con Panel
tabs_monitor = pn.Tabs(
#    ('Plant', svg_pane),
    ('Charts', plot_pane)
)
tabs_ctrl = pn.Tabs(
    ('Enviroment', pn.Column(
        sld_f_x_label,sld_f_x))
  #  ('POI', pn.widgets.Button(name='Set to 2', button_type='success')),
)

# # A reusable function to update the SVG (for buttons and sliders)
# def update_svg(event=None):
#     current_svg = svg_pane.object
    
#     new_radius = slider1.value
#     new_color = color_picker.value
    
#     if event and event.obj is button1:
#         new_color = '#FF0000'
#     elif event and event.obj is button2:
#         new_color = '#00FF00'

#     # Use regular expressions for more reliable attribute updates
#     updated_svg = re.sub(
#         r'(<circle id="interactive_circle".*?)\sfill=".*?"',
#         r'\1 fill="{}"'.format(new_color),
#         current_svg
#     )
#     updated_svg = re.sub(
#         r'(<circle id="interactive_circle".*?)\sr=".*?"',
#         r'\1 r="{}"'.format(new_radius),
#         updated_svg
#     )

#     # s.set_tspan('U_SS1', f"{meas_dict['V_SS1']:5.3f} pu")
#     # s.set_tspan('U_SS2', f"{meas_dict['V_SS2']:5.3f} pu")
#     # s.set_tspan('U_POI', f"{slider_val:5.3f} pu")
#     # s.set_title('GRID_title', f"V = {meas_dict['V_GRID']:5.3f} pu\nP = {meas_dict['p_line_POIHV_GRID'] / 1e6:5.3f} MW\nQ = {meas_dict['q_line_POIHV_GRID'] / 1e6:5.3f} Mvar")
#     # s.set_tspan('P_POI', f"{meas_dict['p_line_POI_POIHV'] / 1e6:5.1f} MW")
#     # s.set_tspan('Q_POI', f"{meas_dict['q_line_POI_POIHV'] / 1e6:5.1f} Mvar")

    
#     svg_pane.object = updated_svg

# The asynchronous function to be called periodically
async def update_time_label():
    """
    Asynchronously updates the time label in the SVG.
    """
    current_time = datetime.datetime.now().strftime("%H:%M:%S")
    # current_svg = svg_pane.object

    # # Use a regular expression to update the content of the text element
    # new_svg = re.sub(
    #     r'(<tspan id="P_POI".*?>)(.*?)(</tspan>)',
    #     r'\1Time: {}\3'.format(current_time),
    #     current_svg,
    #     flags=re.DOTALL
    # )

    url = "http://localhost:8000/measurements"

    ppc_ip = '10.30.0.4'

    # http://10.30.0.4:5500/ppc_measurements
     
    ##############################################################################
    url = "http://localhost:8000/measurements"
    #url = f"http://{ppc_ip}:5500/ppc_measurements"
    response = requests.get(url, headers={"Content-Type": "application/json"}, timeout=0.5) # Added a timeout
    response.raise_for_status()
    meas_dict = response.json()
    time.sleep(0.05)

    ###############################################################################
    url = "http://localhost:8000/setpoints"
    #url = f"http://{ppc_ip}:5500/ppc_setpoints"
    headers = {"Content-Type": "application/json"}
    data = {
        "f_x":float(sld_f_x.value)
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    
    # # This is the line that's failing
    # response.raise_for_status() 

    #response = requests.post(url, headers=headers, data=json.dumps(data))



    # time.sleep(1.0)

    # # Generate the new SVG
    # s = svg('cuerva_emec_db.svg')
    # s.set_tspan('U_SS1', f"{meas_dict['U_SS1']/20e3:5.3f} pu")
    # s.set_tspan('U_SS2', f"{meas_dict['U_SS2']/20e3:5.3f} pu")
    # s.set_tspan('U_POI', f"{meas_dict['U_POI']/20e3:5.3f} pu")
    # #s.set_title('GRID_title', f"V = {meas_dict['V_GRID']:5.3f} pu\nP = {meas_dict['p_line_POIHV_GRID'] / 1e6:5.3f} MW\nQ = {meas_dict['q_line_POIHV_GRID'] / 1e6:5.3f} Mvar")
    # s.set_tspan('P_POI', f"{meas_dict['p_line_POI_POIHV'] / 1e6:5.2f} MW")
    # s.set_tspan('Q_POI', f"{meas_dict['q_line_POI_POIHV'] / 1e6:5.2f} Mvar")
    
    charts_data['Times'][-1] = time.time() - t_0
    charts_data['theta'][-1] = meas_dict['theta'] 
    charts_data['f_x'][-1] = meas_dict['f_x'] 

    charts_data['Times'][0:-1] = charts_data['Times'][1:]
    charts_data['theta'][0:-1] = charts_data['theta'][1:]
    charts_data['f_x'][0:-1] = charts_data['f_x'][1:]

    charts_data['k'] += 1

    plt.close('all') # Clear previous plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(charts_data['Times'], charts_data['theta'], color='blue') # Plot new random data
    ax.plot(charts_data['Times'], charts_data['f_x'], color='red') # Plot new random data

    plot_pane.object = fig # Assign the new figure to the pane


# # Link widgets to the update function
# button1.on_click(update_svg)
# button2.on_click(update_svg)
# slider1.param.watch(update_svg, 'value')
# color_picker.param.watch(update_svg, 'value')

# Combine all components into a Panel layout
dashboard = pn.Column(
        tabs_monitor,
        tabs_ctrl
)

# Use pn.io.periodic_async to run the update_time_label function every 1000ms (1 second)
# The `dashboard.servable()` is required to use periodic functions in a standalone app.
# In a Jupyter notebook, `dashboard` will automatically display the periodic updates.
pn.state.add_periodic_callback(update_time_label, period=100)

# Display the dashboard
dashboard.show()