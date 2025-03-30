#!/bin/bash

# Exit on error
set -e

# Activate virtual environment
source venv/bin/activate

# Run linting
echo "Running linting..."
flake8 f1_mcp_server tests

# Run type checking
echo "Running type checking..."
mypy f1_mcp_server

# Run tests with coverage
echo "Running tests with coverage..."
pytest tests/ --cov=f1_mcp_server --cov-report=html --cov-report=term-missing

# Run security checks
echo "Running security checks..."
bandit -r f1_mcp_server

echo "All tests completed!" 