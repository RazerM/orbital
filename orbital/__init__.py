from orbital.elements import KeplerianElements


def test():
    """Run tests from orbital/tests directory."""
    import os.path
    import pytest
    pytest.main(os.path.dirname(os.path.abspath(__file__)))
