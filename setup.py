from setuptools import setup, find_packages

setup(
    name="denoising-diffusion-deep-fake",
    version="0.1.0",
    packages=find_packages(include=["d3f"]),
    entry_points={
        "console_scripts":[
            "d3f = d3f.main:cli"
        ]
    },
)
