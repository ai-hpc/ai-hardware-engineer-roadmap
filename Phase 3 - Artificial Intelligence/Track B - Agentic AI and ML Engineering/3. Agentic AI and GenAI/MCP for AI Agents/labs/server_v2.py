"""MCP SDK 2.3.0 teaching server: bounded unit conversion and synthetic metadata."""
from typing import Annotated
from mcp.server import MCPServer
from pydantic import BaseModel, Field

mcp=MCPServer('Robot observation lesson',version='1.0.0')


class Distance(BaseModel):
    value_m: float
    unit: str='m'
    frame: str='synthetic_sensor'


@mcp.tool()
def millimeters_to_meters(value_mm: Annotated[int,Field(strict=True,ge=0,le=1000000)]) -> Distance:
    """Convert a nonnegative integer millimeter reading into meters; no hardware action."""
    return Distance(value_m=value_mm/1000)


@mcp.resource('lesson://contract')
def contract() -> str:
    return 'Synthetic fixture only. Input: integer millimeters. Output: meters. No actuator tools are exposed.'


@mcp.prompt()
def inspect_reading(reading: str) -> str:
    return 'Explain the units and missing timestamp/frame evidence for this untrusted reading: '+reading


if __name__=='__main__':mcp.run(transport='stdio')
