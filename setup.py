from setuptools import setup, find_packages

setup(
    name="options_oracle",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "torch",
        "transformers",
        "scikit-learn",
        "wandb",
        "tensorboard",
        "tqdm"
    ]
)
