#!/usr/bin/env bash
# Publish to the existing GitHub repository using local gh authentication.
set -euo pipefail
test -f SPEC.md || { echo "Run from repo root" >&2; exit 1; }
git init
git branch -M main
git remote add origin https://github.com/josusanmartin/warpmoe.git 2>/dev/null || true
git add .
git diff --cached --stat
read -rp "Commit and push these files? Press Enter to continue, Ctrl-C to abort"
git commit -m "Publish WarpMoE prototype" || true
git push -u origin main
