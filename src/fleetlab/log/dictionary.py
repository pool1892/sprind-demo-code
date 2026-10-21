"""The data dictionary the agents receive, generated from the visible part of the log schema.

The tables and columns come from `contracts.log.data_dictionary_markdown`,
which reads the columns tagged visible; this module adds the conventions a
reader needs to line the tables up. Everything here is shown to the agents.
"""

from __future__ import annotations

from fleetlab.contracts.log import data_dictionary_markdown

__all__ = ["CONVENTIONS", "DICTIONARY_FILE", "data_dictionary"]

# Where the dictionary lies in an export, next to `log/`.
DICTIONARY_FILE = "data_dictionary.md"

CONVENTIONS = """## Conventions

- Time is robot time. Control runs at 20 steps per second, so `steps.time_s`
  is the step number times 0.05 s; `episodes.started_at` is the time of day
  the episode started and `episodes.day` counts days from 0.
- `steps` has one row per state of the robot. Row 0 is the state after the
  episode's reset, row k the state after k actions. `commanded_action` is the
  action the robot was commanded in that state (position, rotation, gripper);
  the last row has none. `episodes.steps` is the number of actions, which is
  also the last row's step.
- `steps.frame_index` numbers the frames of the episode's videos. Both videos
  have a frame at the same steps; a step without a frame has it empty. Use
  the frame index, not the time, to find a frame.
- The videos are H.264 and therefore compressed: their pixel values differ
  slightly from the images the cameras delivered. The scene camera is fixed
  in the work cell, the wrist camera moves with the gripper.
- Identifiers carry no meaning: `e000123` is an episode, `r07` a robot, `s2`
  a site; `fleet` in a scope means all robots.
- A vector column such as `joint_pos` (`double[7]`) is a list in SQL, and
  DuckDB indexes list elements from 1: `joint_pos[1]` is the first joint.
- `change_log.old` and `change_log.new` hold the value as JSON text.
"""


def data_dictionary() -> str:
    """The data dictionary as Markdown: every visible table and column, and the conventions."""
    return data_dictionary_markdown().rstrip() + "\n\n" + CONVENTIONS
