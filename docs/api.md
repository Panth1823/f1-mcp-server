# F1 MCP API Documentation

## Overview

The F1 MCP API provides access to Formula 1 racing data through a standardized interface. All endpoints are prefixed with `/mcp/function/`.

## Authentication

Currently, the API does not require authentication.

## Endpoints

### Get MCP Context

```http
GET /mcp/context
```

Returns information about the MCP server's capabilities and available functions.

### Get Current Session

```http
GET /mcp/function/get_current_session
```

Returns information about the current or most recent F1 session.

### Get Driver Standings

```http
GET /mcp/function/get_driver_standings
```

Parameters:
- `year` (optional): Year to get standings for. Defaults to current year.

Returns the current Formula 1 Driver's Championship standings.

### Get Constructor Standings

```http
GET /mcp/function/get_constructor_standings
```

Parameters:
- `year` (optional): Year to get standings for. Defaults to current year.

Returns the current Formula 1 Constructor's Championship standings.

### Get Race Calendar

```http
GET /mcp/function/get_race_calendar
```

Parameters:
- `year` (required): Year to get calendar for.

Returns the Formula 1 race calendar for the specified year.

### Get Session Results

```http
GET /mcp/function/get_session_results
```

Parameters:
- `year` (required): Year of the event
- `event` (required): Event identifier (e.g., "bahrain", "monaco")
- `session` (required): Session type ("FP1", "FP2", "FP3", "Q", "Sprint", "Race")

Returns detailed results for the specified session.

### Get Driver Performance

```http
GET /mcp/function/get_driver_performance
```

Parameters:
- `year` (required): Year of the event
- `event` (required): Event identifier
- `driver` (required): Driver identifier

Returns detailed performance data for a specific driver.

### Get Telemetry

```http
GET /mcp/function/get_telemetry
```

Parameters:
- `year` (required): Year of the event
- `event` (required): Event identifier
- `driver` (required): Driver identifier
- `lap` (required): Lap number

Returns detailed telemetry data for a specific lap.

### Get Weather Data

```http
GET /mcp/function/get_weather_data
```

Parameters:
- `year` (required): Year of the event
- `event` (required): Event identifier

Returns weather information for a specific event.

### Get Circuit Info

```http
GET /mcp/function/get_circuit_info
```

Parameters:
- `circuit_id` (required): Circuit identifier

Returns detailed information about a specific circuit.

## Response Formats

All responses are in JSON format. A typical successful response has this structure:

```json
{
    "status": "success",
    "data": {
        // Response data here
    }
}
```

Error responses have this structure:

```json
{
    "status": "error",
    "message": "Error description"
}
```

## Rate Limiting

Currently, there are no rate limits implemented.

## WebSocket Support

The server also provides real-time updates through WebSocket connections at `/ws/{client_id}`.

## Error Codes

- 400: Bad Request - Invalid parameters
- 404: Not Found - Resource not found
- 500: Internal Server Error - Server-side error

## Best Practices

1. Cache responses when appropriate
2. Use specific parameters rather than fetching all data
3. Handle rate limits gracefully
4. Implement proper error handling 