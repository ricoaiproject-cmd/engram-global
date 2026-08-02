"""engram - a human-like memory MCP server.

A memory model that separates meaning (embedding) from recallability (activation):
- The embedding vector is fixed; search ranking is modulated by ACT-R activation
- Activation rises with use and decays by a power law when neglected (but never vanishes)
- Memories from a striking context (high importance) are encoded more deeply and decay more slowly
- A corrected error is re-encoded at high activation, mistake and all, as its own experience
"""

__version__ = "0.11.0"
