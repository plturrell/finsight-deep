# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU-accelerated visualization components for research contexts and knowledge graphs
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import asyncio
from enum import Enum

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    from vispy import app, scene
    from vispy.visuals.transforms import STTransform
    VISPY_AVAILABLE = True
except ImportError:
    VISPY_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logger = logging.getLogger(__name__)


class VisualizationType(Enum):
    """Types of visualizations"""
    GRAPH_2D = "graph_2d"
    GRAPH_3D = "graph_3d"
    HEATMAP = "heatmap"
    TIMELINE = "timeline"
    DASHBOARD = "dashboard"
    EMBEDDINGS = "embeddings"


@dataclass
class GraphNode:
    """Node in visualization graph"""
    id: str
    label: str
    position: np.ndarray
    color: str = "#3498db"
    size: float = 10.0
    metadata: Dict[str, Any] = None


@dataclass
class GraphEdge:
    """Edge in visualization graph"""
    source_id: str
    target_id: str
    weight: float = 1.0
    color: str = "#95a5a6"
    metadata: Dict[str, Any] = None


class GPULayoutEngine:
    """GPU-accelerated graph layout engine"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.use_gpu = device == 'cuda' and torch.cuda.is_available()
    
    def force_directed_layout(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        iterations: int = 100,
        k: float = 1.0,
        temperature: float = 0.1
    ) -> Dict[str, np.ndarray]:
        """
        Calculate force-directed layout using GPU acceleration
        
        Args:
            nodes: List of graph nodes
            edges: List of graph edges
            iterations: Number of layout iterations
            k: Spring constant
            temperature: Temperature for annealing
        
        Returns:
            Dictionary mapping node IDs to positions
        """
        n_nodes = len(nodes)
        
        if self.use_gpu and CUPY_AVAILABLE:
            # GPU-accelerated layout
            positions = cp.random.rand(n_nodes, 2) * 100
            forces = cp.zeros((n_nodes, 2))
            
            # Create adjacency matrix
            adj_matrix = cp.zeros((n_nodes, n_nodes))
            node_to_idx = {node.id: i for i, node in enumerate(nodes)}
            
            for edge in edges:
                if edge.source_id in node_to_idx and edge.target_id in node_to_idx:
                    i = node_to_idx[edge.source_id]
                    j = node_to_idx[edge.target_id]
                    adj_matrix[i, j] = edge.weight
                    adj_matrix[j, i] = edge.weight
            
            # Force-directed iterations
            for iteration in range(iterations):
                forces.fill(0)
                
                # Repulsive forces (all pairs)
                for i in range(n_nodes):
                    diff = positions[i] - positions
                    distances = cp.linalg.norm(diff, axis=1)
                    distances[distances == 0] = 1e-6
                    
                    repulsion = k * k / distances
                    forces[i] += cp.sum(
                        diff * repulsion[:, cp.newaxis] / distances[:, cp.newaxis],
                        axis=0
                    )
                
                # Attractive forces (connected nodes)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if adj_matrix[i, j] > 0:
                            diff = positions[j] - positions[i]
                            distance = cp.linalg.norm(diff)
                            if distance > 0:
                                attraction = distance * distance / k
                                forces[i] += diff * attraction / distance
                
                # Update positions
                positions += forces * temperature
                temperature *= 0.95  # Cooling
            
            # Convert back to numpy
            final_positions = cp.asnumpy(positions)
        else:
            # CPU fallback
            positions = np.random.rand(n_nodes, 2) * 100
            forces = np.zeros((n_nodes, 2))
            
            # Similar algorithm but on CPU
            node_to_idx = {node.id: i for i, node in enumerate(nodes)}
            
            for iteration in range(iterations):
                forces.fill(0)
                
                # Calculate forces
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        if i != j:
                            diff = positions[i] - positions[j]
                            distance = np.linalg.norm(diff)
                            if distance > 0:
                                # Repulsive force
                                repulsion = k * k / distance
                                forces[i] += diff * repulsion / distance
                
                # Apply forces and update
                positions += forces * temperature
                temperature *= 0.95
            
            final_positions = positions
        
        # Return as dictionary
        return {
            nodes[i].id: final_positions[i]
            for i in range(n_nodes)
        }


class GPUAcceleratedVisualizer:
    """
    GPU-accelerated visualization for research contexts and knowledge graphs
    """
    def __init__(self, use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.layout_engine = GPULayoutEngine('cuda' if self.use_gpu else 'cpu')
        self.current_figure = None
        
        # Check available backends
        self.backends = {
            'vispy': VISPY_AVAILABLE,
            'plotly': PLOTLY_AVAILABLE,
            'matplotlib': True  # Usually available
        }
        
        # Initialize VisPy canvas if available
        if VISPY_AVAILABLE:
            self.canvas = scene.SceneCanvas(
                keys='interactive',
                show=False,
                bgcolor='white'
            )
            self.view = self.canvas.central_widget.add_view()
        else:
            self.canvas = None
            self.view = None
    
    def visualize_knowledge_graph(
        self,
        graph_data: Dict[str, Any],
        visualization_type: VisualizationType = VisualizationType.GRAPH_2D,
        backend: str = 'auto'
    ) -> Any:
        """
        Visualize a knowledge graph with GPU acceleration
        
        Args:
            graph_data: Dictionary containing nodes and edges
            visualization_type: Type of visualization to create
            backend: Visualization backend ('vispy', 'plotly', 'matplotlib', 'auto')
        
        Returns:
            Visualization object (depends on backend)
        """
        # Extract nodes and edges
        nodes = [
            GraphNode(
                id=n.get('id'),
                label=n.get('label', n.get('id')),
                position=np.array(n.get('position', [0, 0])),
                color=n.get('color', '#3498db'),
                size=n.get('size', 10.0),
                metadata=n.get('metadata', {})
            )
            for n in graph_data.get('nodes', [])
        ]
        
        edges = [
            GraphEdge(
                source_id=e.get('source'),
                target_id=e.get('target'),
                weight=e.get('weight', 1.0),
                color=e.get('color', '#95a5a6'),
                metadata=e.get('metadata', {})
            )
            for e in graph_data.get('edges', [])
        ]
        
        # Calculate layout
        positions = self.layout_engine.force_directed_layout(nodes, edges)
        
        # Update node positions
        for node in nodes:
            node.position = positions[node.id]
        
        # Select backend
        if backend == 'auto':
            if visualization_type == VisualizationType.GRAPH_3D and self.backends['vispy']:
                backend = 'vispy'
            elif self.backends['plotly']:
                backend = 'plotly'
            else:
                backend = 'matplotlib'
        
        # Create visualization
        if backend == 'vispy' and self.backends['vispy']:
            return self._create_vispy_visualization(nodes, edges, visualization_type)
        elif backend == 'plotly' and self.backends['plotly']:
            return self._create_plotly_visualization(nodes, edges, visualization_type)
        else:
            return self._create_matplotlib_visualization(nodes, edges, visualization_type)
    
    def _create_vispy_visualization(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        visualization_type: VisualizationType
    ):
        """Create VisPy visualization"""
        if not VISPY_AVAILABLE or not self.view:
            raise ImportError("VisPy is not available")
        
        # Clear previous visualization
        self.view.scene.children.clear()
        
        # Create node visual
        node_positions = np.array([n.position for n in nodes])
        node_colors = np.array([self._hex_to_rgb(n.color) for n in nodes])
        node_sizes = np.array([n.size for n in nodes])
        
        markers = scene.visuals.Markers(
            pos=np.column_stack([node_positions, np.zeros(len(nodes))]),
            edge_color=None,
            face_color=node_colors,
            size=node_sizes,
            edge_width=0,
            symbol='o',
            parent=self.view.scene
        )
        
        # Create edge visual
        edge_positions = []
        edge_colors = []
        
        node_dict = {n.id: n for n in nodes}
        
        for edge in edges:
            if edge.source_id in node_dict and edge.target_id in node_dict:
                source = node_dict[edge.source_id]
                target = node_dict[edge.target_id]
                edge_positions.extend([
                    np.append(source.position, 0),
                    np.append(target.position, 0)
                ])
                edge_colors.extend([
                    self._hex_to_rgb(edge.color),
                    self._hex_to_rgb(edge.color)
                ])
        
        if edge_positions:
            lines = scene.visuals.Line(
                pos=np.array(edge_positions),
                color=np.array(edge_colors),
                parent=self.view.scene,
                connect='segments'
            )
        
        # Set up camera
        if visualization_type == VisualizationType.GRAPH_3D:
            self.view.camera = scene.TurntableCamera(
                elevation=30,
                azimuth=30,
                fov=60,
                distance=200
            )
        else:
            self.view.camera = scene.PanZoomCamera()
        
        # Show canvas
        self.canvas.show()
        return self.canvas
    
    def _create_plotly_visualization(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        visualization_type: VisualizationType
    ):
        """Create Plotly visualization"""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not available")
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        edge_x = []
        edge_y = []
        
        node_dict = {n.id: n for n in nodes}
        
        for edge in edges:
            if edge.source_id in node_dict and edge.target_id in node_dict:
                source = node_dict[edge.source_id]
                target = node_dict[edge.target_id]
                edge_x.extend([source.position[0], target.position[0], None])
                edge_y.extend([source.position[1], target.position[1], None])
        
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        node_x = [n.position[0] for n in nodes]
        node_y = [n.position[1] for n in nodes]
        node_text = [n.label for n in nodes]
        
        fig.add_trace(go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            marker=dict(
                size=[n.size for n in nodes],
                color=[n.color for n in nodes],
                line_width=2
            ),
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Knowledge Graph Visualization",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
    
    def _create_matplotlib_visualization(
        self,
        nodes: List[GraphNode],
        edges: List[GraphEdge],
        visualization_type: VisualizationType
    ):
        """Create matplotlib visualization"""
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw edges
        node_dict = {n.id: n for n in nodes}
        
        for edge in edges:
            if edge.source_id in node_dict and edge.target_id in node_dict:
                source = node_dict[edge.source_id]
                target = node_dict[edge.target_id]
                ax.plot(
                    [source.position[0], target.position[0]],
                    [source.position[1], target.position[1]],
                    c=edge.color,
                    alpha=0.5,
                    linewidth=edge.weight
                )
        
        # Draw nodes
        for node in nodes:
            ax.scatter(
                node.position[0],
                node.position[1],
                c=node.color,
                s=node.size * 10,
                alpha=0.8,
                edgecolors='white',
                linewidth=2
            )
            ax.text(
                node.position[0],
                node.position[1] + node.size * 0.2,
                node.label,
                ha='center',
                va='bottom',
                fontsize=8
            )
        
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title("Knowledge Graph")
        plt.tight_layout()
        
        return fig
    
    def create_dashboard(
        self,
        data: List[Dict[str, Any]],
        layout: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Create an interactive dashboard with GPU acceleration
        
        Args:
            data: List of visualization data
            layout: Dashboard layout configuration
        
        Returns:
            Dashboard object
        """
        if not PLOTLY_AVAILABLE:
            # Fallback to matplotlib
            import matplotlib.pyplot as plt
            
            n_plots = len(data)
            fig, axes = plt.subplots(
                nrows=(n_plots + 1) // 2,
                ncols=2,
                figsize=(15, 5 * ((n_plots + 1) // 2))
            )
            
            if n_plots == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            for i, plot_data in enumerate(data):
                ax = axes[i]
                plot_type = plot_data.get('type', 'line')
                
                if plot_type == 'line':
                    ax.plot(plot_data['x'], plot_data['y'])
                elif plot_type == 'bar':
                    ax.bar(plot_data['x'], plot_data['y'])
                elif plot_type == 'scatter':
                    ax.scatter(plot_data['x'], plot_data['y'])
                
                ax.set_title(plot_data.get('title', f'Plot {i+1}'))
                ax.set_xlabel(plot_data.get('xlabel', 'X'))
                ax.set_ylabel(plot_data.get('ylabel', 'Y'))
            
            plt.tight_layout()
            return fig
        
        # Create Plotly dashboard
        n_plots = len(data)
        
        # Default layout
        if layout is None:
            cols = 2
            rows = (n_plots + 1) // 2
        else:
            rows = layout.get('rows', 1)
            cols = layout.get('cols', n_plots)
        
        # Create subplots
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[d.get('title', f'Plot {i+1}') for i, d in enumerate(data)]
        )
        
        # Add each plot
        for i, plot_data in enumerate(data):
            row = i // cols + 1
            col = i % cols + 1
            
            plot_type = plot_data.get('type', 'line')
            
            if plot_type == 'line':
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['x'],
                        y=plot_data['y'],
                        mode='lines',
                        name=plot_data.get('name', 'Line')
                    ),
                    row=row,
                    col=col
                )
            elif plot_type == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=plot_data['x'],
                        y=plot_data['y'],
                        name=plot_data.get('name', 'Bar')
                    ),
                    row=row,
                    col=col
                )
            elif plot_type == 'scatter':
                fig.add_trace(
                    go.Scatter(
                        x=plot_data['x'],
                        y=plot_data['y'],
                        mode='markers',
                        name=plot_data.get('name', 'Scatter')
                    ),
                    row=row,
                    col=col
                )
            elif plot_type == 'heatmap':
                fig.add_trace(
                    go.Heatmap(
                        z=plot_data['z'],
                        x=plot_data.get('x'),
                        y=plot_data.get('y'),
                        colorscale='Viridis'
                    ),
                    row=row,
                    col=col
                )
        
        # Update layout
        fig.update_layout(
            height=300 * rows,
            showlegend=True,
            title_text="Research Dashboard"
        )
        
        return fig
    
    def visualize_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: Optional[List[str]] = None,
        method: str = 'tsne',
        n_components: int = 2
    ) -> Any:
        """
        Visualize high-dimensional embeddings
        
        Args:
            embeddings: Tensor of embeddings (n_samples, n_features)
            labels: Optional labels for points
            method: Dimensionality reduction method ('tsne', 'pca', 'umap')
            n_components: Number of components (2 or 3)
        
        Returns:
            Visualization object
        """
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
        
        # Dimensionality reduction
        if method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=n_components)
            reduced = reducer.fit_transform(embeddings_np)
        elif method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=n_components, random_state=42)
            reduced = reducer.fit_transform(embeddings_np)
        elif method == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings_np)
            except ImportError:
                logger.warning("UMAP not available, falling back to t-SNE")
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=n_components, random_state=42)
                reduced = reducer.fit_transform(embeddings_np)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Create visualization
        if PLOTLY_AVAILABLE:
            if n_components == 2:
                fig = go.Figure(data=[go.Scatter(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    mode='markers+text',
                    text=labels if labels else None,
                    textposition="top center",
                    marker=dict(
                        size=8,
                        color=list(range(len(reduced))),
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                
                fig.update_layout(
                    title=f'Embedding Visualization ({method.upper()})',
                    xaxis_title='Component 1',
                    yaxis_title='Component 2'
                )
            else:  # 3D
                fig = go.Figure(data=[go.Scatter3d(
                    x=reduced[:, 0],
                    y=reduced[:, 1],
                    z=reduced[:, 2],
                    mode='markers+text',
                    text=labels if labels else None,
                    marker=dict(
                        size=5,
                        color=list(range(len(reduced))),
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                
                fig.update_layout(
                    title=f'3D Embedding Visualization ({method.upper()})',
                    scene=dict(
                        xaxis_title='Component 1',
                        yaxis_title='Component 2',
                        zaxis_title='Component 3'
                    )
                )
            
            return fig
        else:
            # Matplotlib fallback
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(10, 8))
            
            if n_components == 2:
                plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.7)
                if labels:
                    for i, label in enumerate(labels):
                        plt.annotate(label, (reduced[i, 0], reduced[i, 1]))
                plt.xlabel('Component 1')
                plt.ylabel('Component 2')
            else:  # 3D
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], alpha=0.7)
                if labels:
                    for i, label in enumerate(labels):
                        ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], label)
                ax.set_xlabel('Component 1')
                ax.set_ylabel('Component 2')
                ax.set_zlabel('Component 3')
            
            plt.title(f'Embedding Visualization ({method.upper()})')
            return fig
    
    def _hex_to_rgb(self, hex_color: str) -> np.ndarray:
        """Convert hex color to RGB array"""
        hex_color = hex_color.lstrip('#')
        return np.array([int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)])
    
    def export(self, visualization: Any, format: str, filename: str):
        """
        Export visualization to file
        
        Args:
            visualization: Visualization object
            format: Export format ('png', 'html', 'svg', 'pdf')
            filename: Output filename
        """
        if hasattr(visualization, 'write_image'):  # Plotly
            if format == 'html':
                visualization.write_html(filename)
            else:
                visualization.write_image(filename, format=format)
        elif hasattr(visualization, 'savefig'):  # Matplotlib
            visualization.savefig(filename, format=format, dpi=300, bbox_inches='tight')
        elif hasattr(visualization, 'render'):  # VisPy
            img = visualization.render()
            import imageio
            imageio.imwrite(filename, img)
        else:
            logger.warning(f"Cannot export visualization of type {type(visualization)}")


# Convenience functions
def create_knowledge_graph_visualization(
    graph_data: Dict[str, Any],
    use_gpu: bool = True,
    backend: str = 'auto'
) -> Any:
    """Create a knowledge graph visualization"""
    visualizer = GPUAcceleratedVisualizer(use_gpu=use_gpu)
    return visualizer.visualize_knowledge_graph(graph_data, backend=backend)


def create_research_dashboard(
    data: List[Dict[str, Any]],
    use_gpu: bool = True
) -> Any:
    """Create a research dashboard"""
    visualizer = GPUAcceleratedVisualizer(use_gpu=use_gpu)
    return visualizer.create_dashboard(data)


def visualize_embeddings(
    embeddings: torch.Tensor,
    labels: Optional[List[str]] = None,
    method: str = 'tsne',
    use_gpu: bool = True
) -> Any:
    """Visualize embeddings"""
    visualizer = GPUAcceleratedVisualizer(use_gpu=use_gpu)
    return visualizer.visualize_embeddings(embeddings, labels, method)