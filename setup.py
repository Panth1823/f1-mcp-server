from setuptools import setup, find_packages

setup(
    name="f1_mcp_server",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-dotenv",
        "mcp",
        "fastf1",
        "pandas",
        "numpy",
        "aiohttp",
        "websockets",
        "pydantic",
    ],
) 