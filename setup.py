from setuptools import setup, find_packages

setup(
    name="bridge",
    version="0.1.0",
    description="A brief description of your project",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/bridge",  # Update with your actual repo URL
    packages=find_packages(),  # Automatically find packages in the current directory
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Specify the Python version compatibility
)
