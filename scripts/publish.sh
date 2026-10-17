#!/usr/bin/env bash
# Create public GitHub repo and push v0.1.0. Needs: gh auth login
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$PATH"

if ! gh auth status >/dev/null 2>&1; then
  echo "No GitHub session. One of:"
  echo "  gh auth login -h github.com -p https -w"
  echo "  echo \$GITHUB_TOKEN | gh auth login --with-token"
  exit 2
fi

OWNER="${GITHUB_OWNER:-$(gh api user --jq .login)}"
NAME="${GITHUB_REPO:-raidflash-engine}"
VIS="${GITHUB_VISIBILITY:-public}"

if gh repo view "$OWNER/$NAME" >/dev/null 2>&1; then
  echo "repo exists: $OWNER/$NAME"
else
  gh repo create "$OWNER/$NAME" --"$VIS" --source="$ROOT" --remote=origin --description "RAIDFLASH: one user, every free GPU stream, until it stops scaling"
fi

git push -u origin HEAD
git push origin --tags
echo "https://github.com/$OWNER/$NAME"
