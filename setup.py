from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="portwine",
    version="0.1.5",
    author="Stuart Farmer",
    author_email="stuart@lamden.io",
    description="A clean, elegant portfolio backtester that simplifies strategy development with online processing and comprehensive analysis tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StuartFarmer/portwine",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.2.3",
        "matplotlib>=3.10.1",
        "scipy>=1.15.2",
        "statsmodels>=0.14.4",
        "tqdm>=4.67.1",
        "cvxpy>=1.6.4",
        "fredapi>=0.5.2",
        "pandas-market-calendars>=5.0.0",
        "requests>=2.31.0",
        "pytz>=2024.1",
        "django-scheduler>=0.10.1",
        "flask>=3.1.0",
        "rich>=14.0.0",
        "fastparquet>=2024.2.0",
        "freezegun>=1.5.1",
        "numba>=0.61.2",
        "httpx>=0.28.1",
    ],
)

