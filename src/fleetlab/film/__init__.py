"""The demo's film, its narration and its public site.

`python -m fleetlab.film build` assembles one MP4 from a manifest of beats whose clips are
recorded runs (`manifest`, `sources`), with title cards, burned-in captions and a side
panel (`compose`), narration from `tools/film-narrate` (`narration`), encoded by the ffmpeg
that `imageio_ffmpeg` ships (`encode`); every clip, number and output is listed with its
SHA-256 in `provenance.json` (`build`). `python -m fleetlab.film site` turns the film, the
record page and the saved artifacts into a static site (`website`).

Rendering is CPU work on recorded videos; no simulator or model is loaded, so a build runs
as a light job (`tools/heavy-run --light -m 2G`).
"""
