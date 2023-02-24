from setuptools import setup
  
setup(
    name='spatiotemporal_ensemble_dynamics',
    version='0.1',
    description='Code associated with manuscript',
    author='Guillaume Etter',
    author_email='etterguillaume@gmail.com',
    packages=['models', 'functions','tests'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'h5py',
        'pandas',
        'pyyaml',
        'pingouin',
        'notebook',
        'jupyterlab',
        'matplotlib',
        'tqdm',
        'torch>=1.13.1',
        'torchvision>=0.14.1',
        'umap-learn==0.5',
        'tensorflow-macos',
        'tensorflow-probability'
    ],
)