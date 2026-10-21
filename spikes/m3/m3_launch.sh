#!/usr/bin/env bash
# M3's launch sequence (docs/design/m3-recipe.md, section 9). Staged: it refuses to run
# until M3_LAUNCH=go and every placeholder below is filled from the recorded freeze:
#
#   START   a detached checkout of the starting commit, after `make setup` there
#   C       the starting commit (section 8)
#   DRAW    the recorded draw (seeds and ids; spikes/m3/m3_stage.py's docstring)
#
# Every step leaves an exit receipt `$M3/<step>.exit`; the wrappers return the step's own
# status, and a step runs only when the receipts of what it needs read 0. Heavy steps of
# our own log queued/started/ended lines in the shared queue log; the orchestrators
# (rounds, compare, plain search) log their heavy steps through --queue-log.
set -u

START=${START:-"<START>"}
C=${C:-"<START_COMMIT>"}
DRAW=${DRAW:-"<DRAW.json>"}
M3=${M3:-/var/tmp/robots-loop/m3}
QL=${QL:-/var/tmp/robots-loop/queue.log}
W=${W:-$HOME/.cache/robots-probes/worlds}
UV=${UV:-"uv run --locked"}
HEAVY=${HEAVY:-tools/heavy-run}

# ---- the wrappers ---------------------------------------------------------------------------

heavy() {  # heavy PURPOSE ATTEMPTS EXITFILE COMMAND... : a heavy step of our own
  local p=$1 a=$2 x=$3 s
  shift 3
  echo "$(date -Is) queued lane=m3 commit=$C purpose=$p attempts=$a" >> "$QL"
  $HEAVY sh -c 'echo "$(date -Is) started lane=m3 commit='"$C"' purpose='"$p"' attempts='"$a"'" >> '"$QL"'
    "$@"; s=$?; echo "$(date -Is) ended lane=m3 commit='"$C"' purpose='"$p"' attempts='"$a"' exit=$s" >> '"$QL"'
    exit $s' sh "$@"
  s=$?
  echo "$s" > "$x"
  return "$s"
}

run() {  # run EXITFILE COMMAND... : a CPU step or an orchestrator
  local x=$1 s
  shift
  "$@"
  s=$?
  echo "$s" > "$x"
  return "$s"
}

need() {  # need EXITFILE... : succeed only if every receipt reads 0
  local x
  for x in "$@"; do
    if [ ! -f "$x" ]; then
      echo "m3: $(basename "$x" .exit) was not attempted; what needs it is not run" >&2
      return 1
    fi
    if [ "$(cat "$x")" != 0 ]; then
      echo "m3: $(basename "$x" .exit) did not succeed; what needs it is not run" >&2
      return 1
    fi
  done
}

# ---- the guard ------------------------------------------------------------------------------

staged() {
  [ "${M3_LAUNCH:-}" = go ] || { echo "m3: staged; set M3_LAUNCH=go once the freeze is recorded" >&2; return 2; }
  case "$START$C$DRAW" in *"<"*) echo "m3: a placeholder is still open" >&2; return 2 ;; esac
}

# ---- the sequence ---------------------------------------------------------------------------

main() {
  staged || return $?
  cd "$START" || return 2
  mkdir -p "$M3/tmp"
  export TMPDIR=$M3/tmp OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  local agent="--backend codex --model gpt-6-astra --effort xhigh --service-tier fast"
  local eval="--panel-per-robot-day 12 --panel-video-index 0 --workers 8 --queue-log $QL"
  local stage="$UV python spikes/m3/m3_stage.py"
  local f r id site L library outcome

  # 1. Compose the three fleets (CPU) from the recorded draw: nothing else runs without it.
  run "$M3/compose.exit" $stage compose --draw "$DRAW" > "$M3/compose.json" || return 1

  # 2. Round 1's operation of each fleet, 864 attempts at 8 workers.
  for f in lead variant later; do
    heavy "m3-$f-round1-operation" 864 "$M3/$f-operation.exit" \
      $stage operate --draw "$DRAW" --fleet "$f"
  done

  # 3. The lead runs A, then B (learning on and probes off: the driver's defaults).
  id=$(jq -r .lead.world_id "$DRAW")
  for r in A B; do
    need "$M3/lead-operation.exit" &&
      run "$M3/lead-$r.exit" $UV python -m fleetlab.loop.driver rounds \
        --world-root "$W/$id" --out "$M3/lead-$r" $agent --attempts 480 --max-sessions 4 \
        $eval --label "m3-lead-$r" --fleet-label "the lead fleet"
  done

  # 4. The source (section 4): A's library, else B's, else the recorded 'no-source'. A
  #    failure of this step is an error, not 'no-source': the reuse pair then waits.
  run "$M3/source.exit" $stage source --lead-a "$M3/lead-A" --lead-b "$M3/lead-B" \
    --out "$M3/source.json" > /dev/null
  library=""
  outcome=""
  if need "$M3/source.exit"; then
    library=$(jq -r '.library // empty' "$M3/source.json")
    outcome=$(jq -r '.outcome' "$M3/source.json")
  fi

  # 5. The variant's run.
  id=$(jq -r .variant.world_id "$DRAW")
  need "$M3/variant-operation.exit" &&
    run "$M3/variant.exit" $UV python -m fleetlab.loop.driver rounds \
      --world-root "$W/$id" --out "$M3/variant" $agent --attempts 480 --max-sessions 4 \
      $eval --label m3-variant --fleet-label "the variant fleet"

  # 6. The reuse pair on the later fleet, only with a source; its score after it.
  id=$(jq -r .later.world_id "$DRAW")
  if [ -n "$library" ]; then
    need "$M3/later-operation.exit" &&
      run "$M3/reuse.exit" $UV python -m fleetlab.loop.reuse_run compare \
        --world-root "$W/$id" --library "$library" --out "$M3/reuse" $agent \
        --attempts 480 --max-sessions 4 $eval --order without,with &&
      run "$M3/reuse-score.exit" $UV python -m fleetlab.loop.reuse_run score \
        --comparison "$M3/reuse" --world-root "$W/$id"
  elif [ "$outcome" = no-source ]; then
    echo "m3: no-source (recorded in $M3/source.json): no reuse pair, score or its exhibits" >&2
  fi

  # 7. The plain search per fleet, told the site; any time after its round-1 operation.
  for f in lead variant later; do
    id=$(jq -r ".$f.world_id" "$DRAW")
    site=$($stage site --draw "$DRAW" --fleet "$f") || continue
    need "$M3/$f-operation.exit" &&
      run "$M3/$f-plain.exit" $UV python -m fleetlab.loop.plain_search \
        --world "$W/$id" --out "$M3/$f-plain" --scope "$site" $eval
  done

  # 8. The references per fleet, in their own loop directory, after the fleet's first
  #    engagement decided (the later fleet's only with the reuse pair).
  for r in lead:lead-A variant:variant later:reuse; do
    f=${r%%:*}
    L=$M3/${r#*:}/loop
    [ "$f" = later ] && L=$M3/reuse/engagements/without/loop
    if [ "$f" = later ] && [ "$outcome" = no-source ]; then
      echo "m3: no-source: the later fleet has no engagement to cut its references at" >&2
      continue
    fi
    id=$(jq -r ".$f.world_id" "$DRAW")
    need "$M3/${r#*:}.exit" &&
      run "$M3/$f-references-prep.exit" $stage references-prep \
        --loop "$L" --out "$M3/$f-references" &&
      heavy "m3-$f-references-evaluation" 1152 "$M3/$f-references-evaluation.exit" \
        $stage references-evaluate --world "$W/$id" --loop "$L" --out "$M3/$f-references"
  done

  # 9. The exhibit per attempted run, whatever its outcome. Reporting needs only that
  #    the run was attempted (its receipt exists); a failed or partial run's exhibit is
  #    built from its records with its status, and its film is skipped, with the reason,
  #    where its evaluation is unavailable. The simulation gates above stay as they are.
  for r in lead:lead-A lead:lead-B variant:variant later:reuse/engagements/without \
           later:reuse/engagements/with; do
    f=${r%%:*}
    local run_dir=${r#*:} gate=${r#*:}
    case "$gate" in reuse/*) gate=reuse ;; esac
    local name=${run_dir//\//-}
    if [ ! -f "$M3/$gate.exit" ]; then
      if [ "$gate" = reuse ] && [ "$outcome" = no-source ]; then
        continue  # no-source: the reuse pair was not attempted, as recorded
      fi
      echo "m3: $name was not attempted; it has no exhibit" >&2
      continue
    fi
    site=$($stage site --draw "$DRAW" --fleet "$f") || continue
    run "$M3/exhibit-$name.exit" $UV python -m fleetlab.loop.exhibit build \
      "$M3/$run_dir/engagement/engagement.json" --out "$M3/exhibits/$name" --no-clips
    run "$M3/exhibit-extras-$name.exit" $stage exhibit-extras --run "$M3/$run_dir" \
      --run-exit "$M3/$gate.exit" --build-exit "$M3/exhibit-$name.exit" \
      --site "$site" --bundle "$M3/exhibits/$name"
  done

  # The launcher's own status: non-zero when any step failed, each named.
  local failed=""
  for x in "$M3"/*.exit; do
    [ "$(cat "$x")" = 0 ] || failed="$failed $(basename "$x" .exit)"
  done
  if [ -n "$failed" ]; then
    echo "m3: failed steps:$failed" >&2
    return 1
  fi
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
  exit $?
fi
