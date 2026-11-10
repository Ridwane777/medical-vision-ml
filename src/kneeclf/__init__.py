"""kneeclf: binary knee X-ray classification (healthy vs. osteoarthritis) with classical machine learning.

Only scikit-learn, pandas, numpy, matplotlib, Pillow and the Python standard library are used.
"""
from .train import Config, Result, run

__all__ = ["Config", "Result", "run"]
__version__ = "1.0.0"
