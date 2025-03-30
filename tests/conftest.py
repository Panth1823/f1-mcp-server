import pytest
import json
import os
from pathlib import Path
import aiohttp
import pytest_asyncio

@pytest.fixture
def base_url():
    return "http://localhost:8000/mcp/function"

@pytest.fixture
def test_data():
    return {
        "year": 2024,
        "event": "bahrain",
        "driver": "1",
        "lap": 1
    }

@pytest.fixture
def schema_dir():
    return Path(__file__).parent / "schemas"

@pytest.fixture
def load_schema():
    def _load_schema(schema_name):
        schema_path = Path(__file__).parent / "schemas" / f"{schema_name}.schema.json"
        with open(schema_path) as f:
            return json.load(f)
    return _load_schema

@pytest_asyncio.fixture
async def session():
    async with aiohttp.ClientSession() as session:
        yield session 