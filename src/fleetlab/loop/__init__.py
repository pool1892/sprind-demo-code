"""The two-round improvement loop (docs/design/loop.md).

Lanes: `layout`, `state`, `operate`, `export`, `evaluate` (lane state), `driver` (lane
driver), `learned` (lane learn). Each module is imported by its own name; this package
root imports nothing, so that a module stays usable while another lane's is being built.
"""
