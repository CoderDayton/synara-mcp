"""Multi-process coordination: leader election, discovery, follower proxying.

Each ``synara-mcp`` subprocess Claude spawns runs the same boot flow:

1. Try to acquire an exclusive flock on the leader lockfile.
2. Winner = leader: opens the DB, binds a dynamic HTTP MCP endpoint,
   writes the discovery file, hosts the dashboard.
3. Loser = follower: reads the discovery file and runs a FastMCP proxy
   over its stdio that forwards to the leader's HTTP endpoint.

The kernel releases the flock when the leader's last FD closes (crash
or clean exit), so followers detect death via a separate-FD lock probe
— no heartbeat needed.
"""

from __future__ import annotations
