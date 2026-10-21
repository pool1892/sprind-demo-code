# The demo's film and site

What is here:

- `manifests/`: film manifests (`fleetlab.film-manifest/1`), one per cut. Each beat
  lists its shots; each clip names its recorded run, episode and control steps.
  `rough-2026-10-19.json` is the rough timeline from placeholders.
- `manifests/*.provenance.json`: the provenance of a committed cut (the MP4 itself stays
  under `~/.cache/robots-probes/demo/2026-10-20/film/`).
- `bundles/`: site descriptions (`fleetlab.film-site/1`): which film, which record
  page and which artifacts go on the site, and optionally the site's `base_url` (for
  absolute link previews).
- `narration/script.md`: the narration, one section per beat, with every number a
  placeholder (`{release_delivered}`); `narration/fields.json` binds each placeholder to a
  field of a named record; `manifests/<cut>.records.json` names the files that stand for
  those records in a cut; `narration/voice-test.md`: the passage the candidate voices read.
- `templates/`, `static/`: the site's HTML templates and stylesheet.
- `deploy/railway/`: the two files Railway needs to serve the site (an alternative to
  GitHub Pages).

Builders prepare the site; the coordinator deploys it. The site repository is
`pool1892/sprind-demo-page` (since 19 October): private during preparation, then
public with GitHub Pages. Push the reviewed replacement build
before changing visibility; earlier built versions remain in the public Git history.

## Build the film and the site

From the robots checkout (the environment from `make setup`):

```sh
out=~/.cache/robots-probes/demo/2026-10-20/film

# resolve every clip and count, print the timeline (seconds), render nothing
uv run --locked python -m fleetlab.film check site/manifests/rough-2026-10-19.json

# the film: film.mp4 (1280 x 720, H.264 and AAC, for the web) and film-master.mp4
# (1920 x 1080), captions, chapters, poster, timeline and provenance.json
tools/heavy-run --light -m 2G uv run --locked python -m fleetlab.film build \
    site/manifests/rough-2026-10-19.json --out "$out/rough-2026-10-19" \
    [--narration "$out/narration/<voice id>"] [--story A] [--draft]

# the site, from a site description naming that film folder
uv run --locked python -m fleetlab.film site site/bundles/rough-2026-10-19.json \
    --out "$out/site"
```

Before-and-after pairs are chosen by rule, not by looking: `pairs` matches two runs'
episodes on the same starts, counts rescued and broken, and prints the first rescued
and the first broken start as clips to paste into a `pair` shot:

```sh
uv run --locked python -m fleetlab.film pairs \
    --before <naive chain run> --after <released program run> \
    [--before-where variant=naive] [--after-where variant=release]
```

A `calls` shot lists the agent's tool calls from its session transcript
(`agent-runs/<run>/stdout.jsonl`), optionally of named tools (`"tools": ["query",
"run_check"]`). A manifest names the poster's beat with `"poster": "<beat id>"`. A clip of a jobs run
can use a verified re-render from `python -m fleetlab.jobs replay` with
`"rerender": {"video": ..., "verification": <its report .json>}`; without an accepted
report the original is shown and the build warns.

The site build fails if a file is over 100 MB (GitHub Pages refuses it) and warns above
50 MB. The web film's video bitrate is capped at 3 Mb/s, so a three-minute film stays
under 80 MB.

To look at it locally: `python3 -m http.server 8765 --bind 127.0.0.1 --directory "$out/site"`,
then `tools/film-check-public http://127.0.0.1:8765/`.

## Narration

The script's numbers are bound to the cut's records first; the bound script and its
`bindings.json` (every value with its field and the SHA-256 of its file) go to a folder of
their own, and the manifest's `script` names that bound script. A name left unbound shows
as `{name}` in rehearsal captions; `--final`, a voice recording and `build --final` refuse it.

```sh
uv run --locked python -m fleetlab.film bind site/narration/script.md \
    --records site/manifests/<cut>.records.json --out "$out/<cut>/narration" [--final]
```

```sh
# choose the voice: lists the account's voices and models, picks two premade voices
# labelled male and American suited to narration, and the best model; reads the same
# thirty-second passage with both into $out/voices/ (choices.md says which file is which)
ELEVENLABS_API_KEY=... tools/film-narrate voices

# read the whole script, its numbers bound from the records, into $out/narration/<voice id>/;
# the voice defaults to Brian (nPczCjzI2devNBz1zQrb), --voice replaces it; --candidates
# id,id makes `voices` read the passage with those two instead of the listing's pick
ELEVENLABS_API_KEY=... tools/film-narrate render --records site/manifests/<cut>.records.json \
    [--voice <voice id>]
```

The key is read from the environment only: it reaches curl on standard input, never on a
command line, and is never printed or written to a file. Without the key both commands
run dry: they write the requests they would send and the caption files, and say so.

## Deploy to GitHub Pages (a separate repository holding only the built site)

The site goes into `pool1892/sprind-demo-page`, which holds the built folder and nothing
else: the film, the record page, the clips and the artifacts, no simulator code.
In the commands below, `<repo>` means `sprind-demo-page`.

1. Confirm the existing site repository; do not create a replacement:

   ```sh
   gh repo view pool1892/<repo> --json nameWithOwner,visibility,defaultBranchRef
   ```

2. Clone it once to a folder of its own, outside the robots checkout:

   ```sh
   git clone git@github.com:pool1892/<repo>.git ~/code/<repo>
   ```

3. Copy the built site into the clone, replacing what was there, and push it to the
   default branch:

   ```sh
   rsync -a --delete --exclude .git "$out/site/" ~/code/<repo>/
   cd ~/code/<repo>
   git add -A
   git commit -m "Site: <cut name>, film sha256 $(jq -r .film.sha256 site-files.json | cut -c1-12)"
   git push origin HEAD
   ```

   The folder carries a `.nojekyll` file, so Pages serves every file as it is.

4. At final link handoff, the coordinator makes the repository public, then
   enables Pages from the default branch's root (once):

   ```sh
   gh repo edit pool1892/<repo> --visibility public --accept-visibility-change-consequences
   branch=$(gh repo view pool1892/<repo> --json defaultBranchRef --jq .defaultBranchRef.name)
   gh api -X POST repos/pool1892/<repo>/pages -f "source[branch]=$branch" -f "source[path]=/"
   ```

   Or in the repository's Settings, Pages, Build and deployment: Deploy from a branch,
   that branch, folder `/ (root)`.

5. Wait until the build says `built`, then check from this machine without credentials:

   ```sh
   gh api repos/pool1892/<repo>/pages/builds/latest --jq .status
   ~/code/robots/tools/film-check-public https://pool1892.github.io/<repo>/
   ```

   The check fetches the page and the film with GET and without credentials. The page
   must end in HTTP 200 as HTML, not at a login page, and carry the site build's mark and
   its `<video>`; the film must end in 200 as `video/mp4`, arrive completely within 100 MB
   at the length the server declared, and be an MP4 file. It prints `verdict: PUBLIC`
   (exit 0) or `verdict: NOT PUBLIC` with the reason (exit 1). This is the gate-3 check.

A later cut repeats step 3 and step 5 only.

## Alternatively: Railway

Railway builds a container from the folder it is given; the two files in
`deploy/railway/` make that container Caddy serving the site.

```sh
mkdir -p ~/code/<repo>-railway
rsync -a --delete "$out/site/" ~/code/<repo>-railway/
cp site/deploy/railway/Dockerfile site/deploy/railway/Caddyfile ~/code/<repo>-railway/
cd ~/code/<repo>-railway
railway login            # once, in a browser
railway init             # a new project, named as you like
railway up --detach      # build and deploy this folder
railway domain           # a public *.up.railway.app address
~/code/robots/tools/film-check-public https://<that address>/
```
