def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu: marks tests that require a GPU (deselect with -m \"not gpu\")"
    )
