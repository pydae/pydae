import hjson
import networkx as nx
import matplotlib.pyplot as plt

class PowerSystemVisualizer:
    """
    A class to parse a power system .hjson file and generate a topology diagram
    where spatial distances are proportional to system per-unit reactance.
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = self._load_hjson()
        
        # Extract system base MVA (convert from VA to MVA)
        self.S_base_sys = self.data.get("system", {}).get("S_base", 100e6) / 1e6
        
        # Build dictionaries for quick lookups
        self.buses = {bus["name"]: bus for bus in self.data.get("buses", [])}
        self.lines = self.data.get("lines", [])
        self.syns = {syn["bus"]: syn["name"] for syn in self.data.get("syns", [])}
        
        self.G = nx.Graph()

    def _load_hjson(self):
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return hjson.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find the file: {self.filepath}")
        except Exception as e:
            raise ValueError(f"Error parsing HJSON: {e}")

    def build_graph(self):
        """Constructs the NetworkX graph and calculates dynamic per-unit reactances."""
        for line in self.lines:
            j = line["bus_j"]
            k = line["bus_k"]
            
            if "km" in line:
                # Transmission Line
                km = line["km"]
                X_actual = line["X_km"] * km
                
                # Dynamically fetch the voltage base for this line from bus j
                V_base = self.buses[j]["U_kV"]
                Z_base_sys = (V_base**2) / self.S_base_sys
                
                X_pu_sys = X_actual / Z_base_sys
                label = f"X={X_pu_sys:.4f} pu"
                edge_type = 'line'
            else:
                # Transformer / Generator Step-up
                S_mva = line["S_mva"]
                X_pu_old = line["X_pu"]
                
                # Convert equipment pu to system base pu
                X_pu_sys = X_pu_old * (self.S_base_sys / S_mva)
                label = f"X={X_pu_sys:.4f} pu"
                edge_type = 'trafo'
                
            self.G.add_edge(j, k, X_pu=X_pu_sys, label=label, type=edge_type)
            
            # Ensure nodes are added with their voltage attributes
            self.G.nodes[j]['U_kV'] = self.buses[j]["U_kV"]
            self.G.nodes[k]['U_kV'] = self.buses[k]["U_kV"]

    def draw_diagram(self, output_filename="power_system_diagram.png"):
        """Draws the topology and saves it to an image file."""
        if len(self.G.nodes) == 0:
            self.build_graph()
            
        # Distance proportional to reactance
        for u, v, d in self.G.edges(data=True):
            d['weight'] = 1.0 / (d['X_pu'] + 0.001)

        # Generate Spring Layout
        pos = nx.spring_layout(self.G, weight='weight', seed=42, k=0.8, iterations=1000)

        plt.figure(figsize=(12, 9))

        # Color nodes dynamically based on voltage level
        node_colors = []
        for node in self.G.nodes():
            u_kv = self.G.nodes[node].get('U_kV', 0)
            if u_kv >= 400.0:
                node_colors.append('lightblue')  # High Voltage
            else:
                node_colors.append('lightgreen') # Medium Voltage

        # Draw Base Graph
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors, node_size=800, edgecolors='black')
        
        # Enhance labels by adding Generator names if they exist at that bus
        node_labels = {}
        for node in self.G.nodes():
            label = str(node)
            if node in self.syns:
                label += f"\n({self.syns[node]})"
            node_labels[node] = label
            
        nx.draw_networkx_labels(self.G, pos, labels=node_labels, font_size=11, font_weight='bold')

        # Draw Edges based on type
        lines_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d['type'] == 'line']
        trafo_edges = [(u, v) for u, v, d in self.G.edges(data=True) if d['type'] == 'trafo']

        nx.draw_networkx_edges(self.G, pos, edgelist=lines_edges, width=2, edge_color='gray')
        nx.draw_networkx_edges(self.G, pos, edgelist=trafo_edges, width=2, edge_color='orange', style='dashed')

        # Draw Edge Labels
        edge_labels = nx.get_edge_attributes(self.G, 'label')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels, font_size=8)

        # Plot Styling
        title = (f"Power System Topology [{self.data.get('system', {}).get('name', 'Unknown')}]\n"
                 f"(Distance mapped to System pu Reactance | S_base={self.S_base_sys} MVA)\n"
                 f"Blue: 400 kV | Green: 20 kV (Generators)")
        
        plt.title(title, fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Diagram successfully saved to {output_filename}")

# ==========================================
# Example Usage (Can be run directly)
# ==========================================
if __name__ == "__main__":
    # Ensure you have your HJSON file in the same directory, or provide the full path
    hjson_file = "sys8bus4gen_new.hjson"
    
    try:
        visualizer = PowerSystemVisualizer(hjson_file)
        visualizer.draw_diagram("sys8bus4gen_topology.png")
    except Exception as e:
        print(f"Failed to generate diagram: {e}")