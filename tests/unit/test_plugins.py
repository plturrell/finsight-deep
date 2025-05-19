from aiq.runtime.loader import discover_and_register_plugins, PluginTypes
from aiq.cli.type_registry import GlobalTypeRegistry

# Load plugins
discover_and_register_plugins(PluginTypes.CONFIG_OBJECT)

# Check what was loaded
print("LLM Registry:")
print(GlobalTypeRegistry._llm_registry)
print("\nWorkflow Registry:")
print(GlobalTypeRegistry._workflow_registry)
print("\nFunction Registry:")
print(GlobalTypeRegistry._function_registry)