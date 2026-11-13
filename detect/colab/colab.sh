#!/bin/bash
# Drives the Colab scripts from the Mac. Works from anywhere inside the repo.
# Before the first line: colab new -s train --gpu T4, then colab drivemount -s train (a person, every session).
#
#   detect/colab/colab.sh setup                          the code at HEAD and the data onto the machine
#   detect/colab/colab.sh train run1=train.yaml,run2=run2.yaml [RESUME=1] [KEEP_THE_MACHINE=1]
#   detect/colab/colab.sh log run1 [lines]               the end of a run's log, also while it trains
#   detect/colab/colab.sh fetch run1 [run2 ...]          run folders and logs into runs/, also while the machine trains
#   detect/colab/colab.sh stop                           lets Drive finish writing, then stops the machine. Always, at the
#                                                        end: an idle machine uses up the quota
#
# log and fetch read Drive, so they need no setup and no GPU: colab new -s train, colab drivemount -s train, fetch.
# SESSION=<name> uses another session than "train".
# Run setup again after every new commit: train records HEAD as the commit of the code on the machine.
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

SESSION=${SESSION:-train}
RUNS_ON_DRIVE=/content/drive/MyDrive/capstone/runs
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir -p runs

fetch_log() {
    colab download -s "$SESSION" "$RUNS_ON_DRIVE/$1.log" "runs/$1.log" >/dev/null
}

fetch_folder() {
    mkdir -p "$2"
    colab ls -s "$SESSION" "$1" | while IFS= read -r entry; do
        if [[ "$entry" == */ ]]; then
            fetch_folder "$1/${entry%/}" "$2/${entry%/}"
        else
            colab download -s "$SESSION" "$1/$entry" "$2/$entry" >/dev/null
        fi
    done
}

case "${1:-}" in
setup)
    if [ -n "$(git status --porcelain --untracked-files=no -- detect/src uv.lock)" ]; then
        echo "uncommitted changes in detect/src or uv.lock would not reach Colab (the code tar is made from HEAD): commit them first" >&2
        exit 1
    fi
    git archive -o "$scratch/code.tar" HEAD
    colab upload -s "$SESSION" "$scratch/code.tar" /content/code.tar
    colab exec -s "$SESSION" --timeout 0 -f detect/colab/session.py
    ;;
train)
    settings=(--env "RUNS=$2" --env "COMMIT=$(git rev-parse HEAD)")
    for setting in "${@:3}"; do settings+=(--env "$setting"); done
    caffeinate -is colab exec -s "$SESSION" --timeout 0 "${settings[@]}" -f detect/colab/run.py 2>&1 |
        tee "runs/colab-$(date +%Y%m%d-%H%M%S).log"
    ;;
log)
    fetch_log "$2"
    progress_bars_shown_once='s/\r$//; s/.*\r//; s/\e\[[0-9;]*[A-Za-z]//g'
    perl -pe "$progress_bars_shown_once" "runs/$2.log" | grep -v '^ *$' | tail -n "${3:-30}"
    ;;
fetch)
    for name in "${@:2}"; do
        fetch_folder "$RUNS_ON_DRIVE/$name" "runs/$name"
        echo "fetched runs/$name"
        fetch_log "$name" && echo "fetched runs/$name.log" || echo "no $name.log on Drive: a run cut by Colab loses it" >&2
    done
    ;;
stop)
    echo "from google.colab import drive; drive.flush_and_unmount()" >"$scratch/flush.py"
    colab exec -s "$SESSION" --timeout 120 -f "$scratch/flush.py" || echo "Drive was not told to finish writing" >&2
    colab stop -s "$SESSION"
    ;;
*)
    sed -n '2,14p' "$0" >&2
    exit 1
    ;;
esac
