#!/usr/bin/env bash
# Download the optional 1M-step paper checkpoints into runs/.
#   bash scripts/download_checkpoints.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG="${PGFSPP_RELEASE_TAG:-checkpoints}"
REPO="${PGFSPP_GITHUB_REPO:-bz317/PGFS_plus_plus}"
URL="${PGFSPP_CHECKPOINT_URL:-https://github.com/${REPO}/releases/download/${TAG}/pgfspp_checkpoints.tar.gz}"
DEST="${ROOT}/runs"
ARCHIVE="${DEST}/pgfspp_checkpoints.tar.gz"

mkdir -p "${DEST}"
echo "Downloading ${URL}"
if command -v curl >/dev/null 2>&1; then
  curl -L --fail -o "${ARCHIVE}" "${URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${ARCHIVE}" "${URL}"
else
  echo "Need curl or wget to download checkpoints." >&2
  exit 1
fi

tar -xzf "${ARCHIVE}" -C "${ROOT}"
rm -f "${ARCHIVE}"

echo "Checkpoints extracted under ${DEST}:"
ls -lh "${DEST}"/*/model_step_*.pt
