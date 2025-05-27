from setuptools import setup, find_packages

setup(
    name="emotion_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "scikit-learn",
        "gdown",
        "nltk"
    ],
    entry_points={
        "console_scripts": [
            "inference=inference:main",
        ],
    },
)