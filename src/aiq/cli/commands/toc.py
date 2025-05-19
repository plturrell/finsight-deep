# Documentation Table of Contents Generator

import os
import yaml
from pathlib import Path

def generate_documentation_toc():
    """Generate table of contents for all documentation"""
    
    docs_path = Path("docs/source")
    toc = {
        "Core Components": {
            "Verification System": "workflows/verification/index.md",
            "Nash-Ethereum Consensus": "workflows/consensus/index.md",
            "Research Framework": "workflows/research/index.md",
            "Digital Human": "workflows/digital-human/index.md"
        },
        "Integration Guides": {
            "Overview": "workflows/integration/index.md",
            "Verification-Consensus": "workflows/integration/verification-consensus.md",
            "Knowledge Graph": "workflows/integration/knowledge-graph.md",
            "System Architecture": "workflows/integration/architecture.md"
        },
        "Performance": {
            "GPU Optimization": "workflows/performance/gpu-optimization.md",
            "Benchmarks": "workflows/performance/benchmarks.md"
        },
        "API References": {
            "Verification API": "workflows/verification/api-reference.md",
            "Consensus API": "workflows/consensus/api-reference.md",
            "Research API": "workflows/research/api-reference.md"
        },
        "Examples": {
            "Verification Examples": "workflows/verification/examples.md",
            "Consensus Examples": "workflows/consensus/examples.md",
            "Research Examples": "workflows/research/examples.md"
        }
    }
    
    # Write table of contents
    with open(docs_path / "toc.md", "w") as f:
        f.write("# AIQToolkit Documentation\n\n")
        
        for section, items in toc.items():
            f.write(f"\n## {section}\n\n")
            for title, path in items.items():
                f.write(f"- [{title}]({path})\n")
    
    # Create YAML for Sphinx
    with open(docs_path / "_toc.yml", "w") as f:
        yaml.dump({"format": "jb-book", "chapters": toc}, f)
    
    print("Documentation table of contents generated successfully!")

if __name__ == "__main__":
    generate_documentation_toc()