from setuptools import setup, find_packages

setup(
    name="football-motif-analysis",
    version="0.1.0",
    description="Topological Patterns of Cooperative and Adversarial Interactions in Football Passing Networks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pyarrow>=12.0.0",
        "pyyaml>=6.0",
        "networkx>=3.1",
        "statsmodels>=0.14.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "gnn": ["torch>=2.0.0", "torch_geometric>=2.3.0"],
        "pitch": ["mplsoccer>=1.2.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0"],
    },
)
