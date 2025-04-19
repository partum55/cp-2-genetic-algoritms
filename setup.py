from setuptools import setup, find_packages 
 
setup( 
    name="genetic_ml", 
    version="0.1", 
    packages=find_packages(), 
    install_requires=[ 
        "numpy==2.2", 
        "matplotlib==3.9", 
        "pandas>=2.2.3", 
        "scikit-learn>=1.6.1", 
    ], 
) 
