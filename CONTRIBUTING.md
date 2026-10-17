# Contributing

RAIDFLASH is occupancy for one user. PRs that add multi-tenant "fairness"
that leaves slots idle while a single request runs will be closed.

Bring numbers: `delta prompt_tokens == sum(usage)` or the run is discarded.
P seeds must be drafted (`max_tokens >= 16`). D must not clobber `fnc` with `fc7`.

Python 3.10+, `pytest`. No GPU required for unit tests.
