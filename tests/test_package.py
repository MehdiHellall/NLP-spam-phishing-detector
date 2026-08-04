import importlib

from email_threat_detector import __version__


def test_package_exposes_version() -> None:
    assert __version__ == "0.1.0"


def test_package_modules_import() -> None:
    module_names = [
        "email_threat_detector.cleaning",
        "email_threat_detector.cli",
        "email_threat_detector.constants",
        "email_threat_detector.data",
        "email_threat_detector.duplicates",
        "email_threat_detector.inference",
        "email_threat_detector.metrics",
        "email_threat_detector.models",
        "email_threat_detector.preprocessing",
        "email_threat_detector.sampling",
        "email_threat_detector.splits",
        "email_threat_detector.training",
        "email_threat_detector.transformers",
    ]

    imported = [importlib.import_module(module_name) for module_name in module_names]

    assert [module.__name__ for module in imported] == module_names
