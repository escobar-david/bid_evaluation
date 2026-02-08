from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="bid-evaluation",
    version="0.1.0",
    author="David",  # Change to your name
    author_email="your.email@example.com",  # Change to your email
    description="Open-source bid evaluation library for procurement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bid-evaluation",  # Change username
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Office/Business",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "pyyaml>=5.4.0",
        "openpyxl>=3.0.0",
        "simpleeval>=0.9.13",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "streamlit>=1.20.0",
        ],
    },
)
