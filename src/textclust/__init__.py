"""textclust: adaptive clustering of free-text documents stored in any CSV file.

Only scikit-learn, pandas, numpy, matplotlib and the Python standard library are used.
"""
from .pipeline import Config, Result, run

__all__ = ["Config", "Result", "run"]
__version__ = "1.0.0"
