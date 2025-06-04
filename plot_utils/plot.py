import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def plot_power_grid_v1(graph, node_voltages=None, edge_loadings=None, 
                   pos=None, figsize=(12, 8), 
                   show_node_labels=True, show_edge_labels=True,
                   node_cmap='RdYlGn_r', edge_cmap='plasma',
                   voltage_range=(0.9, 1.1), loading_range=(0, 1),
                   node_size=800, edge_width_scale=3,
                   title="Power System Grid Visualization"):
    """
    Visualize a power system grid with voltage and current loading information.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The power grid topology
    node_voltages : dict or None
        Dictionary mapping node_id -> voltage (per unit)
    edge_loadings : dict or None  
        Dictionary mapping (node1, node2) -> loading (0-1)
    pos : dict or None
        Node positions. If None, uses spring layout
    figsize : tuple
        Figure size (width, height)
    show_node_labels : bool
        Whether to show voltage values on nodes
    show_edge_labels : bool
        Whether to show loading values on edges
    node_cmap : str
        Colormap for node voltages
    edge_cmap : str
        Colormap for edge loadings
    voltage_range : tuple
        (min_voltage, max_voltage) for colormap normalization
    loading_range : tuple
        (min_loading, max_loading) for colormap normalization
    node_size : int
        Size of nodes
    edge_width_scale : float
        Scaling factor for edge width based on loading
    title : str
        Plot title
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Generate layout if not provided
    if pos is None:
        pos = nx.spring_layout(graph, k=2, iterations=50)
    
    # Set up voltage visualization
    if node_voltages is not None:
        voltage_values = [node_voltages.get(node, 1.0) for node in graph.nodes()]
        voltage_norm = Normalize(vmin=voltage_range[0], vmax=voltage_range[1])
        node_colors = voltage_values
    else:
        node_colors = 'lightblue'
        voltage_values = None
    
    # Set up loading visualization  
    if edge_loadings is not None:
        loading_values = []
        edge_widths = []
        for edge in graph.edges():
            loading = edge_loadings.get(edge, edge_loadings.get((edge[1], edge[0]), 0.0))
            loading_values.append(loading)
            edge_widths.append(1 + edge_width_scale * loading)
        
        loading_norm = Normalize(vmin=loading_range[0], vmax=loading_range[1])
        edge_colors = loading_values
    else:
        edge_colors = 'gray'
        edge_widths = 1
        loading_values = None
    
    # Draw nodes
    node_cmap = plt.get_cmap(node_cmap) if node_voltages else None
    nodes = nx.draw_networkx_nodes(graph, pos, 
                                  node_color=node_colors,
                                  node_size=node_size,
                                  cmap=node_cmap,
                                  vmin=voltage_range[0] if node_voltages else None,
                                  vmax=voltage_range[1] if node_voltages else None,
                                  ax=ax)
    
    # Draw edges
    edge_cmap = plt.get_cmap(edge_cmap) if edge_loadings else None
    edges = nx.draw_networkx_edges(graph, pos,
                                  edge_color=edge_colors,
                                  width=edge_widths,
                                  edge_cmap=edge_cmap,
                                  edge_vmin=0.,
                                  edge_vmax=1.,
                                  ax=ax)
    
    # Add node labels (voltage values)
    if show_node_labels and node_voltages is not None:
        node_labels = {node: f"{voltage:.3f}" for node, voltage in node_voltages.items()}
        nx.draw_networkx_labels(graph, pos, node_labels, font_size=8, ax=ax)
    else:
        # Show node IDs
        nx.draw_networkx_labels(graph, pos, ax=ax)
    
    # Add edge labels (loading values)
    if show_edge_labels and edge_loadings is not None:
        edge_labels = {}
        for edge in graph.edges():
            loading = edge_loadings.get(edge, edge_loadings.get((edge[1], edge[0]), 0.0))
            edge_labels[edge] = f"{loading:.2f}"
        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=7, ax=ax)
    
    # Add colorbars
    if node_voltages is not None:
        # Voltage colorbar
        sm_voltage = ScalarMappable(norm=voltage_norm, cmap=node_cmap)
        sm_voltage.set_array([])
        cbar_voltage = plt.colorbar(sm_voltage, ax=ax, shrink=0.6, pad=0.1)
        cbar_voltage.set_label('Voltage (p.u.)', rotation=270, labelpad=15)
    
    if edge_loadings is not None:
        # Loading colorbar  
        sm_loading = ScalarMappable(norm=loading_norm, cmap=edge_cmap)
        sm_loading.set_array([])
        cbar_loading = plt.colorbar(sm_loading, ax=ax, shrink=0.6, pad=0.15)
        cbar_loading.set_label('Loading (%)', rotation=270, labelpad=15)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax


def plot_power_grid(
    graph,
    voltage_display='both',      # Options: 'text', 'color', 'both', 'none'
    loading_display='both',      # Options: 'text', 'color', 'both', 'none'
    voltage_key='voltage',       # The key for voltage data in node attributes
    loading_key='loading',       # The key for loading data in edge attributes
    figsize=(12, 10),            # Figure size
    node_size=300,               # Base node size
    edge_width=2,                # Base edge width
    font_size=8,                 # Font size for labels
    voltage_cmap='coolwarm',     # Colormap for voltage
    loading_cmap='plasma',       # Colormap for loading
    voltage_vmin=0.9,            # Min voltage for colormap scaling
    voltage_vmax=1.1,            # Max voltage for colormap scaling
    loading_vmin=0,              # Min loading for colormap scaling
    loading_vmax=1,              # Max loading for colormap scaling
    show_colorbar=True,          # Whether to show colorbars
    node_type_key='type',        # Optional: key for node type (generator, load, etc.)
    show_node_types=False,       # Whether to use different node shapes for types
    node_shapes={                # Shapes for different node types
        'generator': 's',        # Square
        'load': 'o',             # Circle
        'bus': 'o',              # Circle
        'default': 'o'           # Default shape
    },
    pos=None,                    # Optional node positions
    show_node_ids=True,          # Whether to show node IDs
    **kwargs                     # Additional arguments for nx.draw
):
    """
    Visualizes a power system grid with voltage at nodes and loading at edges.
    
    Parameters:
    -----------
    graph : networkx.Graph
        The power system graph to visualize
    voltage_display : str
        How to display voltage: 'text', 'color', 'both', or 'none'
    loading_display : str
        How to display loading: 'text', 'color', 'both', or 'none'
    voltage_key : str
        The key for voltage data in node attributes
    loading_key : str
        The key for loading data in edge attributes
    figsize : tuple
        Figure size
    node_size : int
        Base node size
    edge_width : int
        Base edge width
    font_size : int
        Font size for labels
    voltage_cmap : str
        Colormap for voltage
    loading_cmap : str
        Colormap for loading
    voltage_vmin : float
        Min voltage for colormap scaling
    voltage_vmax : float
        Max voltage for colormap scaling
    loading_vmin : float
        Min loading for colormap scaling
    loading_vmax : float
        Max loading for colormap scaling
    show_colorbar : bool
        Whether to show colorbars
    node_type_key : str
        The key for node type in node attributes
    show_node_types : bool
        Whether to use different node shapes for types
    node_shapes : dict
        Shapes for different node types
    pos : dict
        Node positions (if None, spring_layout will be used)
    show_node_ids : bool
        Whether to show node IDs as labels (True) even when voltage is shown as color
    **kwargs
        Additional arguments for nx.draw
        
    Returns:
    --------
    fig, ax : tuple
        The figure and axis objects
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions if not provided
    if pos is None:
        pos = nx.spring_layout(graph)
    
    # Process node colors based on voltage
    node_colors = []
    if voltage_display in ['color', 'both']:
        voltage_cmap_obj = cm.get_cmap(voltage_cmap)
        for node in graph.nodes():
            voltage = graph.nodes[node].get(voltage_key, 1.0)  # Default to 1.0 if not specified
            # Normalize voltage value to colormap range
            norm_voltage = (voltage - voltage_vmin) / (voltage_vmax - voltage_vmin)
            norm_voltage = max(0, min(1, norm_voltage))  # Clamp to [0, 1]
            node_colors.append(voltage_cmap_obj(norm_voltage))
    else:
        if show_node_types:
            for node in graph.nodes():
                node_type = graph.nodes[node].get(node_type_key, 'default')
                color = 'lightblue'  # Default color
                if node_type == 'generator':
                    color = 'green'
                elif node_type == 'load':
                    color = 'red'
                node_colors.append(color)
        else:
            node_colors = ['lightblue' for _ in graph.nodes()]
    
    # Process edge colors based on loading
    edge_colors = []
    edge_widths = []
    if loading_display in ['color', 'both']:
        loading_cmap_obj = cm.get_cmap(loading_cmap)
        for u, v, data in graph.edges(data=True):
            loading = data.get(loading_key, 0.0)  # Default to 0.0 if not specified
            # Normalize loading value to colormap range
            norm_loading = (loading - loading_vmin) / (loading_vmax - loading_vmin)
            norm_loading = max(0, min(1, norm_loading))  # Clamp to [0, 1]
            edge_colors.append(loading_cmap_obj(norm_loading))
            # Optionally scale edge width based on loading
            edge_widths.append(edge_width * (1 + loading))
    else:
        edge_colors = ['gray' for _ in graph.edges()]
        edge_widths = [edge_width for _ in graph.edges()]
    
    # Draw the network
    if show_node_types:
        # Create node lists by type
        node_lists = {}
        for node in graph.nodes():
            node_type = graph.nodes[node].get(node_type_key, 'default')
            if node_type not in node_lists:
                node_lists[node_type] = []
            node_lists[node_type].append(node)
        
        # Draw nodes by type with appropriate shapes
        for node_type, nodes in node_lists.items():
            if nodes:
                node_color_subset = [node_colors[list(graph.nodes()).index(n)] for n in nodes]
                nx.draw_networkx_nodes(
                    graph, pos,
                    nodelist=nodes,
                    node_color=node_color_subset,
                    node_size=node_size,
                    node_shape=node_shapes.get(node_type, node_shapes['default']),
                    ax=ax
                )
    else:
        # Draw all nodes with the same shape
        nx.draw_networkx_nodes(
            graph, pos,
            node_color=node_colors,
            node_size=node_size,
            ax=ax
        )
    
    # Draw edges
    for i, (u, v) in enumerate(graph.edges()):
        nx.draw_networkx_edges(
            graph, pos,
            edgelist=[(u, v)],
            edge_color=[edge_colors[i]],
            width=edge_widths[i],
            ax=ax
        )
    
    # Add node labels for voltage
    if voltage_display in ['text', 'both']:
        # Show voltage values as labels
        node_labels = {}
        for node in graph.nodes():
            voltage = graph.nodes[node].get(voltage_key, 1.0)
            node_labels[node] = f'{voltage:.3f}'
        
        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=font_size,
            ax=ax
        )
    elif voltage_display == 'color' and show_node_ids:
        # Show node IDs as labels
        node_labels = {node: str(node) for node in graph.nodes()}
        
        nx.draw_networkx_labels(
            graph, pos,
            labels=node_labels,
            font_size=font_size,
            ax=ax
        )
    
    # Add edge labels for loading
    if loading_display in ['text', 'both']:
        edge_labels = {}
        for u, v, data in graph.edges(data=True):
            loading = data.get(loading_key, 0.0)
            edge_labels[(u, v)] = f'{loading:.2f}'
        
        nx.draw_networkx_edge_labels(
            graph, pos,
            edge_labels=edge_labels,
            font_size=font_size,
            ax=ax
        )
    
    # Add colorbars if using colors and colorbar is enabled
    if show_colorbar:
        if voltage_display in ['color', 'both']:
            voltage_sm = cm.ScalarMappable(
                cmap=voltage_cmap,
                norm=mcolors.Normalize(vmin=voltage_vmin, vmax=voltage_vmax)
            )
            voltage_sm.set_array([])
            voltage_cbar = plt.colorbar(voltage_sm, ax=ax, location='right', shrink=0.8)
            voltage_cbar.set_label('Voltage (p.u.)')
        
        if loading_display in ['color', 'both']:
            loading_sm = cm.ScalarMappable(
                cmap=loading_cmap,
                norm=mcolors.Normalize(vmin=loading_vmin, vmax=loading_vmax)
            )
            loading_sm.set_array([])
            # Position the loading colorbar
            cbar_position = 'right' if voltage_display not in ['color', 'both'] else None
            loading_cbar = plt.colorbar(loading_sm, ax=ax, location=cbar_position, shrink=0.8, pad=0.1)
            loading_cbar.set_label('Loading')
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax


# Example usage and test function
def create_example_grid():
    """Create an example power grid for demonstration."""
    # Create a simple grid topology
    G = nx.Graph()
    
    # Add nodes (buses)
    buses = [1, 2, 3, 4, 5, 6]
    G.add_nodes_from(buses)
    
    # Add edges (transmission lines)
    lines = [(1, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5), (4, 6), (5, 6)]
    G.add_edges_from(lines)
    
    # Example voltage data (per unit)
    voltages = {1: 1.05, 2: 1.02, 3: 0.98, 4: 0.96, 5: 0.94, 6: 0.92}
    
    # Example loading data (percentage, 0-1)
    loadings = {
        (1, 2): 0.75, (1, 3): 0.60, (2, 3): 0.45,
        (2, 4): 0.80, (3, 5): 0.55, (4, 5): 0.70,
        (4, 6): 0.85, (5, 6): 0.40
    }
    
    return G, voltages, loadings


if __name__ == "__main__":
    # Demonstration
    grid, voltages, loadings = create_example_grid()
    
    # Create visualization
    fig, ax = plot_power_grid(grid, voltages, loadings,
                             title="IEEE Test System - Voltage and Loading")
    plt.show()
    
    # Alternative visualization with custom settings
    fig2, ax2 = plot_power_grid(grid, voltages, loadings,
                               show_edge_labels=False,
                               node_cmap='viridis',
                               edge_cmap='Reds',
                               voltage_range=(0.9, 1.1),
                               title="Custom Color Scheme")
    plt.show()