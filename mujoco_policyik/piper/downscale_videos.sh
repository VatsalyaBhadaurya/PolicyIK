#!/usr/bin/env bash
# Downscale Piper videos ~4x for fast ACT training on a 4050.
# cam_global 1280x720 -> 320x180, cam_gripper 640x480 -> 320x240.
set -e
source ~/lerobot-venv/bin/activate
pip install -q imageio-ffmpeg 2>&1 | tail -1
FF=$(python -c "import imageio_ffmpeg,sys; sys.stdout.write(imageio_ffmpeg.get_ffmpeg_exe())")
echo "ffmpeg: $FF"
SRC=$HOME/piper_train; DST=$HOME/piper_train_s
rm -rf "$DST"; mkdir -p "$DST"
cp -r "$SRC/data" "$SRC/meta" "$DST/"
n=0
for cam in cam_global cam_gripper; do
  mkdir -p "$DST/videos/observation.images.$cam/chunk-000"
  if [ "$cam" = cam_global ]; then SC=320:180; else SC=320:240; fi
  for f in "$SRC/videos/observation.images.$cam/chunk-000/"*.mp4; do
    out="$DST/videos/observation.images.$cam/chunk-000/$(basename "$f")"
    "$FF" -y -i "$f" -vf scale=$SC -r 30 -c:v libx264 -preset veryfast -g 30 \
          -pix_fmt yuv420p -an "$out" -loglevel error &
    n=$((n+1)); (( n % 6 == 0 )) && wait
  done
done
wait
python - <<'PY'
import json, os
p = os.path.expanduser('~/piper_train_s/meta/info.json')
info = json.load(open(p))
g = info['features']['observation.images.cam_global']
g['shape'] = [180, 320, 3]; g['info'].update({'video.height': 180, 'video.width': 320})
r = info['features']['observation.images.cam_gripper']
r['shape'] = [240, 320, 3]; r['info'].update({'video.height': 240, 'video.width': 320})
json.dump(info, open(p, 'w'), indent=1)
print('info.json image shapes -> global 180x320, gripper 240x320')
PY
echo "=== sizes ==="; du -sh "$SRC/videos" "$DST/videos"
echo DOWNSCALE_DONE
