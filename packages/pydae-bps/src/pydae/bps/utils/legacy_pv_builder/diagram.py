import json
import networkx as nx
import matplotlib.pyplot as plt

# Load the network configuration data
with open("copv.json", "r") as f:
    data = json.load(f)

G = nx.Graph()
pos = {}

# Add nodes and extract their positions
for bus in data.get('buses', []):
    name = bus.get('name')
    G.add_node(name)
    if 'pos_x' in bus and 'pos_y' in bus:
        # Flip the x-coordinate left-right by negating it
        pos[name] = (-bus['pos_x'], bus['pos_y'])

# Handle buses missing positions (POI, POIMV, GRID)
missing_pos = [n for n in G.nodes() if n not in pos]
for i, n in enumerate(missing_pos):
    pos[n] = (0, i * 20 - 20) 

# Add edges representing power lines
for line in data.get('lines', []):
    G.add_edge(line['bus_j'], line['bus_k'])

# Add edges representing transformers
for tf in data.get('transformers', []):
    G.add_edge(tf['bus_j'], tf['bus_k'])

# Draw the diagram
plt.figure(figsize=(14, 10))
nx.draw(
    G, 
    pos, 
    with_labels=True, 
    node_size=300, 
    node_color='lightgreen', # Changed color to easily distinguish from the original
    font_size=8, 
    font_weight='bold', 
    edge_color='gray'
)
plt.title('Bus Network Diagram (Left-Right Flipped)')

# Save to file
plt.savefig('bus_network_flipped.png', bbox_inches='tight')