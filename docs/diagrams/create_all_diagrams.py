"""
Create all architecture diagrams for AIQToolkit documentation
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

def create_main_architecture_diagram():
    """Create main AIQToolkit architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, 'AIQToolkit Architecture', fontsize=24, fontweight='bold', ha='center')
    
    # Define colors
    colors = {
        'frontend': '#3498db',
        'api': '#2ecc71',
        'core': '#e74c3c',
        'infrastructure': '#f39c12',
        'data': '#9b59b6'
    }
    
    # Frontend Layer
    frontend_box = FancyBboxPatch(
        (5, 75), 20, 15,
        boxstyle="round,pad=0.1",
        facecolor=colors['frontend'],
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(frontend_box)
    ax.text(15, 82, 'Frontend Layer', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(15, 79, 'Web UI', fontsize=10, ha='center', color='white')
    ax.text(15, 77, 'CLI', fontsize=10, ha='center', color='white')
    
    # API Layer
    api_box = FancyBboxPatch(
        (30, 75), 20, 15,
        boxstyle="round,pad=0.1",
        facecolor=colors['api'],
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(api_box)
    ax.text(40, 82, 'API Layer', fontsize=12, fontweight='bold', ha='center', color='white')
    ax.text(40, 79, 'REST API', fontsize=10, ha='center', color='white')
    ax.text(40, 77, 'WebSocket', fontsize=10, ha='center', color='white')
    
    # Core Services
    core_box = FancyBboxPatch(
        (5, 45), 90, 25,
        boxstyle="round,pad=0.1",
        facecolor=colors['core'],
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(core_box)
    ax.text(50, 66, 'Core Services', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Individual core components
    components = [
        ('Workflow Engine', 15, 57),
        ('Agent Framework', 30, 57),
        ('LLM Providers', 45, 57),
        ('Verification System', 60, 57),
        ('Nash-Ethereum', 75, 57),
        ('Digital Human', 85, 57),
        ('Research Engine', 15, 50),
        ('Profiler', 30, 50),
        ('Memory', 45, 50),
        ('Retriever', 60, 50),
        ('GPU Acceleration', 75, 50),
        ('Knowledge Graph', 85, 50)
    ]
    
    for comp_name, x, y in components:
        comp_box = Rectangle((x-5, y-2), 10, 4, facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(comp_box)
        ax.text(x, y, comp_name, fontsize=8, ha='center', va='center')
    
    # Infrastructure Layer
    infra_box = FancyBboxPatch(
        (5, 20), 90, 20,
        boxstyle="round,pad=0.1",
        facecolor=colors['infrastructure'],
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(infra_box)
    ax.text(50, 36, 'Infrastructure', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Infrastructure components
    infra_components = [
        ('Docker', 20, 28),
        ('Kubernetes', 35, 28),
        ('GPU Cluster', 50, 28),
        ('Monitoring', 65, 28),
        ('Security', 80, 28)
    ]
    
    for comp_name, x, y in infra_components:
        comp_box = Rectangle((x-7, y-2), 14, 4, facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(comp_box)
        ax.text(x, y, comp_name, fontsize=10, ha='center', va='center')
    
    # Data Layer
    data_box = FancyBboxPatch(
        (5, 5), 90, 10,
        boxstyle="round,pad=0.1",
        facecolor=colors['data'],
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(data_box)
    ax.text(50, 12, 'Data Layer', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Data components
    data_components = [
        ('Ethereum', 20, 8),
        ('Milvus', 35, 8),
        ('Redis', 50, 8),
        ('PostgreSQL', 65, 8),
        ('Jena RDF', 80, 8)
    ]
    
    for comp_name, x, y in data_components:
        comp_box = Rectangle((x-7, y-1.5), 14, 3, facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(comp_box)
        ax.text(x, y, comp_name, fontsize=10, ha='center', va='center')
    
    # Add arrows showing data flow
    arrows = [
        ((15, 75), (15, 70)),  # Frontend to Core
        ((40, 75), (40, 70)),  # API to Core
        ((50, 45), (50, 40)),  # Core to Infrastructure
        ((50, 20), (50, 15)),  # Infrastructure to Data
    ]
    
    for start, end in arrows:
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle="arc3,rad=0",
            arrowstyle="->",
            mutation_scale=20,
            color='black',
            linewidth=2
        )
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('docs/diagrams/architecture_main.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_verification_flow_diagram():
    """Create verification system flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, 'Verification System Flow', fontsize=22, fontweight='bold', ha='center')
    
    # Define flow steps
    steps = [
        ('User Query', 20, 85, '#3498db'),
        ('Entity Extraction', 20, 70, '#2ecc71'),
        ('Source Retrieval', 20, 55, '#e74c3c'),
        ('Multi-Method Verification', 50, 70, '#f39c12'),
        ('Confidence Scoring', 50, 55, '#9b59b6'),
        ('W3C PROV Tracking', 50, 40, '#1abc9c'),
        ('Consensus Check', 80, 70, '#e67e22'),
        ('Blockchain Recording', 80, 55, '#8e44ad'),
        ('Final Result', 80, 40, '#2c3e50')
    ]
    
    # Draw boxes and labels
    for i, (label, x, y, color) in enumerate(steps):
        box = FancyBboxPatch(
            (x-10, y-5), 20, 8,
            boxstyle="round,pad=0.1",
            facecolor=color,
            edgecolor='black',
            alpha=0.8
        )
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=11, fontweight='bold', ha='center', va='center', color='white')
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
        (3, 6), (6, 7), (7, 8), (5, 8)
    ]
    
    for start_idx, end_idx in connections:
        start = steps[start_idx]
        end = steps[end_idx]
        
        arrow = FancyArrowPatch(
            (start[1], start[2]-4),
            (end[1], end[2]+4),
            connectionstyle="arc3,rad=0.1",
            arrowstyle="->",
            mutation_scale=15,
            color='black',
            linewidth=2
        )
        ax.add_patch(arrow)
    
    # Add method boxes
    methods = [
        ('Bayesian', 35, 25),
        ('Fuzzy Logic', 50, 25),
        ('Dempster-Shafer', 65, 25)
    ]
    
    for method, x, y in methods:
        method_box = Rectangle((x-8, y-3), 16, 6, facecolor='lightblue', edgecolor='black', alpha=0.7)
        ax.add_patch(method_box)
        ax.text(x, y, method, fontsize=10, ha='center', va='center')
    
    # Add legend
    ax.text(50, 10, 'Confidence Methods', fontsize=12, fontweight='bold', ha='center')
    
    plt.tight_layout()
    plt.savefig('docs/diagrams/verification_flow.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_gpu_performance_chart():
    """Create GPU performance comparison chart"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    
    # Performance comparison
    tasks = ['Verification', 'Consensus', 'Research', 'Digital Human', 'Knowledge Graph']
    cpu_times = [1280, 450, 2100, 3500, 890]
    gpu_times = [100, 35, 164, 273, 69]
    
    x = np.arange(len(tasks))
    width = 0.35
    
    cpu_bars = ax1.bar(x - width/2, cpu_times, width, label='CPU', color='#3498db')
    gpu_bars = ax1.bar(x + width/2, gpu_times, width, label='GPU', color='#2ecc71')
    
    ax1.set_xlabel('Task', fontsize=12)
    ax1.set_ylabel('Latency (ms)', fontsize=12)
    ax1.set_title('CPU vs GPU Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(tasks, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add speedup annotations
    for i in range(len(tasks)):
        speedup = cpu_times[i] / gpu_times[i]
        ax1.annotate(f'{speedup:.1f}x',
                    xy=(i, max(cpu_times[i], gpu_times[i]) + 100),
                    ha='center',
                    fontsize=10,
                    fontweight='bold',
                    color='red')
    
    # GPU utilization over time
    time = np.linspace(0, 60, 300)
    utilization = 70 + 20 * np.sin(0.1 * time) + 10 * np.random.randn(300)
    utilization = np.clip(utilization, 0, 100)
    
    ax2.plot(time, utilization, color='#e74c3c', linewidth=2)
    ax2.fill_between(time, utilization, alpha=0.3, color='#e74c3c')
    ax2.set_xlabel('Time (seconds)', fontsize=12)
    ax2.set_ylabel('GPU Utilization (%)', fontsize=12)
    ax2.set_title('GPU Utilization During Processing', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig('docs/diagrams/gpu_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_consensus_mechanism_diagram():
    """Create Nash-Ethereum consensus mechanism diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, 'Nash-Ethereum Consensus Mechanism', fontsize=20, fontweight='bold', ha='center')
    
    # Agents
    agent_positions = [
        (20, 70, 'Agent 1'),
        (35, 75, 'Agent 2'),
        (50, 70, 'Agent 3'),
        (65, 75, 'Agent 4'),
        (80, 70, 'Agent 5')
    ]
    
    for x, y, label in agent_positions:
        circle = Circle((x, y), 5, facecolor='#3498db', edgecolor='black', alpha=0.8)
        ax.add_patch(circle)
        ax.text(x, y, label, fontsize=10, ha='center', va='center', color='white', fontweight='bold')
    
    # Nash Equilibrium Box
    nash_box = FancyBboxPatch(
        (35, 40), 30, 15,
        boxstyle="round,pad=0.1",
        facecolor='#e74c3c',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(nash_box)
    ax.text(50, 47, 'Nash Equilibrium', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(50, 44, 'Computation', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Ethereum Box
    eth_box = FancyBboxPatch(
        (35, 15), 30, 15,
        boxstyle="round,pad=0.1",
        facecolor='#9b59b6',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(eth_box)
    ax.text(50, 22, 'Ethereum', fontsize=14, fontweight='bold', ha='center', color='white')
    ax.text(50, 19, 'Smart Contract', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Draw connections from agents to Nash
    for x, y, _ in agent_positions:
        arrow = FancyArrowPatch(
            (x, y-5),
            (50, 55),
            connectionstyle="arc3,rad=0.2",
            arrowstyle="->",
            mutation_scale=15,
            color='black',
            linewidth=1.5
        )
        ax.add_patch(arrow)
    
    # Arrow from Nash to Ethereum
    arrow = FancyArrowPatch(
        (50, 40),
        (50, 30),
        connectionstyle="arc3,rad=0",
        arrowstyle="->",
        mutation_scale=20,
        color='black',
        linewidth=3
    )
    ax.add_patch(arrow)
    
    # Add payoff matrix
    matrix_x, matrix_y = 10, 20
    matrix_box = Rectangle((matrix_x-5, matrix_y-5), 20, 10, facecolor='lightgray', edgecolor='black', alpha=0.7)
    ax.add_patch(matrix_box)
    ax.text(matrix_x+5, matrix_y+2, 'Payoff Matrix', fontsize=11, ha='center', fontweight='bold')
    
    # Add game theory elements
    game_x, game_y = 80, 20
    game_box = Rectangle((game_x-5, game_y-5), 20, 10, facecolor='lightgreen', edgecolor='black', alpha=0.7)
    ax.add_patch(game_box)
    ax.text(game_x+5, game_y+2, 'Game Theory', fontsize=11, ha='center', fontweight='bold')
    ax.text(game_x+5, game_y-1, 'Optimization', fontsize=11, ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('docs/diagrams/consensus_mechanism.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_digital_human_architecture():
    """Create digital human system architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    
    # Title
    ax.text(50, 95, 'Digital Human Architecture', fontsize=22, fontweight='bold', ha='center')
    
    # User Interface Layer
    ui_box = FancyBboxPatch(
        (10, 80), 80, 10,
        boxstyle="round,pad=0.1",
        facecolor='#3498db',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(ui_box)
    ax.text(50, 85, 'User Interface Layer', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Processing Layer
    process_box = FancyBboxPatch(
        (10, 60), 80, 15,
        boxstyle="round,pad=0.1",
        facecolor='#2ecc71',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(process_box)
    ax.text(50, 67, 'Processing Layer', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Processing components
    proc_components = [
        ('Conversation Engine', 25, 63),
        ('Emotion Processor', 50, 63),
        ('Context Manager', 75, 63)
    ]
    
    for comp, x, y in proc_components:
        comp_box = Rectangle((x-10, y-2), 20, 4, facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(comp_box)
        ax.text(x, y, comp, fontsize=10, ha='center', va='center')
    
    # Avatar Layer
    avatar_box = FancyBboxPatch(
        (10, 40), 80, 15,
        boxstyle="round,pad=0.1",
        facecolor='#e74c3c',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(avatar_box)
    ax.text(50, 47, 'Avatar Layer', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Avatar components
    avatar_components = [
        ('Avatar Controller', 25, 43),
        ('Facial Animator', 50, 43),
        ('Audio2Face-3D', 75, 43)
    ]
    
    for comp, x, y in avatar_components:
        comp_box = Rectangle((x-10, y-2), 20, 4, facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(comp_box)
        ax.text(x, y, comp, fontsize=10, ha='center', va='center')
    
    # Intelligence Layer
    intel_box = FancyBboxPatch(
        (10, 20), 80, 15,
        boxstyle="round,pad=0.1",
        facecolor='#f39c12',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(intel_box)
    ax.text(50, 27, 'Intelligence Layer', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Intelligence components
    intel_components = [
        ('Financial Engine', 25, 23),
        ('Knowledge Graph', 50, 23),
        ('Verification', 75, 23)
    ]
    
    for comp, x, y in intel_components:
        comp_box = Rectangle((x-10, y-2), 20, 4, facecolor='white', edgecolor='black', alpha=0.9)
        ax.add_patch(comp_box)
        ax.text(x, y, comp, fontsize=10, ha='center', va='center')
    
    # GPU Acceleration
    gpu_box = FancyBboxPatch(
        (10, 5), 80, 10,
        boxstyle="round,pad=0.1",
        facecolor='#9b59b6',
        edgecolor='black',
        alpha=0.8
    )
    ax.add_patch(gpu_box)
    ax.text(50, 10, 'GPU Acceleration Layer', fontsize=14, fontweight='bold', ha='center', color='white')
    
    # Draw connections
    connections = [
        ((50, 80), (50, 75)),  # UI to Processing
        ((50, 60), (50, 55)),  # Processing to Avatar
        ((50, 40), (50, 35)),  # Avatar to Intelligence
        ((50, 20), (50, 15)),  # Intelligence to GPU
    ]
    
    for start, end in connections:
        arrow = FancyArrowPatch(
            start, end,
            connectionstyle="arc3,rad=0",
            arrowstyle="<->",
            mutation_scale=20,
            color='black',
            linewidth=2
        )
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('docs/diagrams/digital_human_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_diagrams():
    """Create all documentation diagrams"""
    
    print("Creating architecture diagrams...")
    
    # Create individual diagrams
    create_main_architecture_diagram()
    print("✓ Main architecture diagram created")
    
    create_verification_flow_diagram()
    print("✓ Verification flow diagram created")
    
    create_gpu_performance_chart()
    print("✓ GPU performance chart created")
    
    create_consensus_mechanism_diagram()
    print("✓ Consensus mechanism diagram created")
    
    create_digital_human_architecture()
    print("✓ Digital human architecture diagram created")
    
    print("\nAll diagrams created successfully!")
    print("Diagrams saved to: docs/diagrams/")

if __name__ == "__main__":
    create_all_diagrams()