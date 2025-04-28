from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="MLOps Implementation with Astro Airflow, Redis and Prometheus",
    version="0.1",
    author="Renswick Delvar",
    author_email="renswick.delver@gmail.com",
    packages=find_packages(),
    install_requires = requirements,
)