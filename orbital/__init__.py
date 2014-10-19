from orbital.elements import KeplerianElements

def test():
    import os.path
    import pytest
    pytest.main(os.path.dirname(os.path.abspath(__file__)))
