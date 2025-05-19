#!/usr/bin/env python
"""Simple test script to verify imports work."""

def test_imports():
    print("Testing core imports...")
    try:
        import aiq
        print("✓ aiq module imported")
        
        from aiq.builder import Builder
        print("✓ Builder imported")
        
        from aiq.runtime import Runner, Session
        print("✓ Runner and Session imported")
        
        from aiq.builder.context import AIQContext
        print("✓ AIQContext imported")
        
        from aiq.builder.workflow import Workflow
        print("✓ Workflow imported")
        
        print("\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


if __name__ == "__main__":
    success = test_imports()
    exit(0 if success else 1)