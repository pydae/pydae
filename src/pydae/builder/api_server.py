import time
import threading
from typing import Dict
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel

# Import the Model class we built previously
from model_class import Model

# --- 1. SETUP APP AND SHARED STATE ---
app = FastAPI(title="DAE Real-Time Digital Twin API")

model = Model()
model_lock = threading.Lock() # Prevents memory tearing between threads

# Buffer to hold incoming API setpoints until the next simulation step
setpoints_buffer: Dict[str, float] = {}

# --- 2. DEFINE JSON DATA STRUCTURES ---
class Setpoints(BaseModel):
    inputs: Dict[str, float]

# --- 3. HTTP API ENDPOINTS ---
@app.post("/api/u")
def update_setpoints(data: Setpoints):
    """
    Accepts JSON setpoints and buffers them for the u vector.
    Example payload: {"inputs": {"f_x": 10.5, "v_ref": 1.0}}
    """
    global setpoints_buffer
    with model_lock:
        # Update the buffer with new incoming values
        setpoints_buffer.update(data.inputs)
        
    return {"status": "success", "buffered_inputs": list(data.inputs.keys())}

@app.get("/api/z")
def read_measurements():
    """
    Reads the current measurements (z vector) from the running model.
    Returns a JSON dictionary of {name: value}.
    """
    with model_lock:
        # Map the variable names in h_list to their current numerical values in z
        measurements = {name: float(model.z[i]) for i, name in enumerate(model.h_list)}
        
        # You can also expose dynamic states (x) or algebraic states (y) here!
        # Example: measurements['theta'] = float(model.get_value('theta'))
        
    return {"time": model.t, "measurements": measurements}

@app.get("/api/status")
def get_status():
    """Health check endpoint."""
    return {"status": "running", "sim_time": model.t}

# --- 4. REAL-TIME SIMULATION ENGINE ---
def simulation_loop():
    global setpoints_buffer
    
    # Initialize the model
    model.Dt = 0.01  # 10ms internal integration step
    with model_lock:
        model.ini({}, xy_0=1) # Adjust initial guess as needed for your specific model
    
    rt_step = 0.05  # Python loop syncs every 50ms
    current_sim_time = 0.0
    
    print("Background simulation loop started.")
    
    while True:
        loop_start = time.perf_counter()
        
        with model_lock:
            # 1. Inject buffered API setpoints into the model
            model.step(current_sim_time + rt_step, setpoints_buffer)
            
            # 2. Advance time
            current_sim_time += rt_step
            
            # 3. Clear the buffer (inputs stay active in u_run automatically)
            setpoints_buffer.clear() 
        
        # 4. Synchronize with Wall-Clock time
        elapsed = time.perf_counter() - loop_start
        time_to_sleep = rt_step - elapsed
        
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        else:
            print(f"Warning: Overrun! Step took {elapsed*1000:.1f}ms")

# --- 5. STARTUP LOGIC ---
@app.on_event("startup")
def startup_event():
    """Starts the simulation thread exactly when the web server starts."""
    sim_thread = threading.Thread(target=simulation_loop, daemon=True)
    sim_thread.start()

if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=False)