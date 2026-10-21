from __future__ import annotations

import pytest

from fleetlab.film import narration

SCRIPT = """# Narration

Text before the first beat is not read.

## open: Opening

This is a robot. It never changes.

<!-- depends: nothing -->
> stage direction, not read

## a1: The skills

One sentence that is long enough to need more than two caption lines when it is wrapped
at forty-two characters, with a clause, and another clause, and then some more words.
"""


def test_parse_keeps_spoken_text_only() -> None:
    beats = narration.parse_script(SCRIPT)
    assert [b.id for b in beats] == ["open", "a1"]
    assert beats[0].text == "This is a robot. It never changes."
    assert beats[0].notes == ("depends: nothing",)
    assert beats[0].sentences == ["This is a robot.", "It never changes."]


def test_duplicate_beat_is_an_error() -> None:
    with pytest.raises(ValueError, match="twice"):
        narration.parse_script("## a: A\nx.\n## a: B\ny.\n")


def test_spoken_seconds_at_reading_pace() -> None:
    beat = narration.parse_script(SCRIPT)[0]
    # 7 words at 150 a minute, and a pause after each of the two sentences
    assert narration.spoken_seconds(beat) == pytest.approx(7 * 60 / 150 + 2 * 0.3)


def test_cues_cover_the_length_in_two_lines_at_most() -> None:
    beat = narration.parse_script(SCRIPT)[1]
    cues = narration.cues(beat, 10.0, offset=5.0)
    assert len(cues) >= 2
    assert cues[0].start == 5.0
    assert cues[-1].end == pytest.approx(15.0, abs=0.01)
    assert all(len(c.text.split("\n")) <= 2 for c in cues)
    assert " ".join(c.text.replace("\n", " ") for c in cues) == beat.text
    text = narration.vtt(cues)
    assert text.startswith("WEBVTT\n\n1\n00:00:05.000 --> ")


def test_request_bodies_fall_back_to_plainer_ones() -> None:
    open_, a1 = narration.parse_script(SCRIPT)
    bodies = narration.request_bodies(a1, model="eleven_v4", previous=open_)
    assert bodies[0]["previous_text"] == open_.text
    assert bodies[0]["model_id"] == "eleven_v4"
    assert "previous_text" not in bodies[1] and "seed" not in bodies[1]
    assert bodies[-1] == {"text": a1.text, "model_id": "eleven_v4"}
    assert all("key" not in str(b).lower() for b in bodies)


def _voice(
    voice_id: str, gender: str, accent: str, use: str = "", category: str = "premade"
) -> dict[str, object]:
    labels = {"gender": gender, "accent": accent, **({"use_case": use} if use else {})}
    return {"voice_id": voice_id, "name": voice_id.upper(), "category": category, "labels": labels}


def test_pick_voices_by_labels_and_suitability() -> None:
    listing = {
        "voices": [
            _voice("f", "female", "american"),
            _voice("uk", "male", "british"),
            _voice("game", "male", "american", "characters, video games"),
            _voice("plain", "male", "american"),
            _voice("nar", "male", "American", "narration"),
            _voice("clone", "male", "american", "narration", category="cloned"),
        ]
    }
    picked = narration.pick_voices(listing)
    assert [v["voice_id"] for v in picked] == ["nar", "plain"]


def test_pick_model_prefers_v4_then_multilingual() -> None:
    assert narration.pick_model(None) == "eleven_multilingual_v2"
    listed = [{"model_id": "eleven_flash_v2_5"}, {"model_id": "eleven_multilingual_v2"}]
    assert narration.pick_model(listed) == "eleven_multilingual_v2"
    assert narration.pick_model([*listed, {"model_id": "eleven_v4"}]) == "eleven_v4"
    assert narration.pick_model([{"model_id": "other"}]) == "eleven_multilingual_v2"


def test_the_repository_script_parses() -> None:
    from fleetlab.film.website import SITE_ROOT

    beats = narration.load_script(SITE_ROOT / "narration" / "script.md")
    assert beats and all(b.text for b in beats)
    assert not any(ch.isdigit() for b in beats for ch in b.text), "no numbers in the narration"
