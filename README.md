# F1 MCP Server

A Formula 1 Machine Control Protocol (MCP) server that provides real-time and historical F1 racing data through a standardized API interface.

## Features

- Real-time session data
- Historical race data
- Driver telemetry
- Weather information
- Championship standings
- Circuit information
- WebSocket support for live updates

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/f1-mcp.git
cd f1-mcp
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Usage

Start the server:

```bash
python -m f1_mcp_server.main
```

The server will start on `http://localhost:8000` by default.

## API Documentation

### Available Endpoints

- `/mcp/context` - Get MCP context information
- `/mcp/function/get_current_session` - Get current session information
- `/mcp/function/get_driver_standings` - Get driver championship standings
- `/mcp/function/get_constructor_standings` - Get constructor championship standings
- `/mcp/function/get_race_calendar` - Get race calendar
- `/mcp/function/get_session_results` - Get session results
- `/mcp/function/get_driver_performance` - Get driver performance data
- `/mcp/function/get_telemetry` - Get detailed telemetry data
- `/mcp/function/get_weather_data` - Get weather information
- `/mcp/function/get_circuit_info` - Get circuit information

For detailed API documentation, see [docs/api.md](docs/api.md).

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
f1_mcp/
├── f1_mcp_server/       # Main package
├── tests/              # Test suite
├── docs/              # Documentation
└── scripts/           # Utility scripts
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request 