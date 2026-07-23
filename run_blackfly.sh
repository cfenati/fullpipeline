#!/usr/bin/env bash
# Load Ubuntu 22.04 FFmpeg libs required by Spinnaker on Ubuntu 24.04.
SPINNAKER_FFMPEG_LIB="${HOME}/spinnaker-ffmpeg-libs/usr/lib/x86_64-linux-gnu"
if [[ -d "${SPINNAKER_FFMPEG_LIB}" ]]; then
  export LD_LIBRARY_PATH="${SPINNAKER_FFMPEG_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "${SCRIPT_DIR}/view_blackfly.py" "$@"
