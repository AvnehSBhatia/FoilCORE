#!/usr/bin/env bash
# Build the FMTS paper. Requires pdflatex + bibtex.
# Usage: bash paper/fmts/build.sh [fmts|fmts_submitted]   (default: fmts = camera-ready)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export TEXINPUTS="${ROOT}/..:${TEXINPUTS:-}"
name="${1:-fmts}"
pdflatex -interaction=nonstopmode "${name}.tex" >"/tmp/${name}.build1.log" || true
bibtex "${name}" >"/tmp/${name}.bibtex.log" || true
pdflatex -interaction=nonstopmode "${name}.tex" >"/tmp/${name}.build2.log" || true
pdflatex -interaction=nonstopmode "${name}.tex" >"/tmp/${name}.build3.log" || true
grep -E "Output written" "/tmp/${name}.build3.log" || echo "no output, see /tmp/${name}.build3.log"
grep -E "^!" "/tmp/${name}.build3.log" || true
