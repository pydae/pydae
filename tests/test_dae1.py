# tests/test_dynamic_module.py

import os
import importlib.util
import pytest

@pytest.fixture
def create_module():
    module_content = """
def dynamic_function(x, y):
    return x * y
"""
    module_path = os.path.join(os.path.dirname(__file__), "dynamic_module.py")

    with open(module_path, "w") as module_file:
        module_file.write(module_content)

    yield module_path

    # Clean up after the test
    #os.remove(module_path)

def test_create_module(create_module):
    assert os.path.isfile(create_module)

def test_dynamic_function(create_module):
    # Import the module dynamically
    module_name = "dynamic_module"
    spec = importlib.util.spec_from_file_location(module_name, create_module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Call the dynamically created function
    result = module.dynamic_function(2, 3)
    assert result == 6
