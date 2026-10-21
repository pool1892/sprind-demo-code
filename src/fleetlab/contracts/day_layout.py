"""The simulated working day's start times (contracts, section 4): public layout.

`identity.started_at_for` turns an episode index into a start time with these
constants, and re-exports them. They live in a module of their own, with no
imports, so that the tool server, which may not import `identity`, can state a
robot-day's capacity for experiment arms.
"""

# The simulated working day: a shift from 08:00 for eight hours.
SHIFT_START_S = 8 * 3600
SHIFT_LENGTH_S = 8 * 3600
# The latest start time of the day, 23:59:59; later indices are refused.
LAST_START_S = 24 * 3600 - 1
# Experiment-arm episodes (indices past the as-run count) from 16:00, one every
# three minutes: 160 slots per robot and day.
ARM_START_S = 16 * 3600
ARM_SPACING_S = 180
ARM_SLOTS = (LAST_START_S - ARM_START_S) // ARM_SPACING_S + 1
