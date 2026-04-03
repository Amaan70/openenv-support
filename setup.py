from setuptools import setup, find_packages

setup(
    name="openenv-customer-support",
    version="1.0.0",
    description="Customer Support Ticket Triage Environment for OpenEnv",
    author="CodeWithAman01",
    packages=find_packages(),
    install_requires=[
        "fastapi==0.104.1",
        "uvicorn==0.24.0",
        "pydantic==2.5.0",
        "openai==1.6.1",
        "requests==2.31.0",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "openenv-server=app:app",
            "openenv-inference=inference:main",
        ],
    },
)