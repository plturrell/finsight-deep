# Basic pytest configuration
import sys
import os

# Add src to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
