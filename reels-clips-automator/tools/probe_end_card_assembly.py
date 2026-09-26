#!/usr/bin/env python3
"""Probe the teaser + main + end-card assembly for boundary health.

Story C2 of docs/tasks/EPIC-content-complete-engaging-clips.md. Instagram
freezes the published reel on the last main frame and never renders the end
card. Desktop players show nothing wrong, because the failure is in the
*timestamps*, not in a decode error.

This script rebuilds the three pieces with the SAME encoder settings
reelsfy.py uses, concatenates them the same way
`assemble_hook_teaser_final` does, and reports whether the boundaries are
healthy. It is a diagnostic, not part of the publish path: run it after
changing any encoder setting, and re-run it for the C4 A/B decision.

    python3 reels-clips-automator/tools/probe_end_card_assembly.py
    python3 reels-clips-automator/tools/probe_end_card_assembly.py --keep

Exits 0 when every boundary is healthy, 1 otherwise, so it can gate a
future CI check.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

# The settings under test. These mirror reelsfy.py; when you change one
# there, change it here too or the probe stops describing the real pipeline.
NVENC = [
    "-c:v", "h264_nvenc", "-profile:v", "high", "-rc:v", "vbr",
    "-qp", "18", "-b:v", "10000k", "-maxrate:v", "12000k",
    "-bufsize:v", "15000k", "-pix_fmt", "yuv420p",
]
FONT = "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"
TEASER_SECONDS = 4
MAIN_SECONDS = 40
FPS = 30


def run(args):
    """Run a command, raising with ffmpeg's stderr when it fails."""
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed (rc={result.returncode}): {' '.join(args)}\n"
            f"{result.stderr[-2000:]}"
        )
    return result


def build_source(path):
    """A 90 s 1080p clip with tone audio - the OBS replay-buffer shape."""
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=s=1920x1080:r={FPS}:d=90",
        "-f", "lavfi", "-i", f"sine=frequency=300:r=48000:d=90",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", path,
    ])


def build_cropped(source, out, seconds):
    """The 9:16 crop, the way generate_short does it (note the -r 30)."""
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", source, "-t", str(seconds),
        "-vf", "crop=608:1080:656:0,scale=1080:1920,setsar=1",
        "-r", str(FPS), "-pix_fmt", "yuv420p", *NVENC,
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", out,
    ])


def burn(input_path, out, label, text):
    """The overlay burn + audio mastering, the way the teaser/main burns do."""
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
        "-i", input_path,
        "-vf",
        f"drawtext=text='{text}':expansion=none:fontfile={FONT}:fontsize=64:"
        "fontcolor=white:borderw=3:bordercolor=black:"
        f"x=(w-tw)/2:y=(h-th)/2:{label}",
        "-af", "loudnorm=I=-14:LRA=11:tp=-1.5,"
               "acompressor=threshold=0.05:ratio=3:makeup=1.2",
        *NVENC, "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-video_track_timescale", "15360", out,
    ])


def build_end_card(out, pin_rate=True):
    """The end card, the way build_end_card_command() builds it.

    `pin_rate` toggles the C3 fix: build_end_card_command() has no -r, so the
    lavfi color source defaults to 25 fps with a 1/12800 timebase, while every
    other piece is 30 fps / 1/15360.
    """
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
        "-f", "lavfi", "-i", "color=c=0x14213D:s=1080x1920:d=1.5",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
    ]
    if pin_rate:
        args += ["-r", str(FPS), "-video_track_timescale", "15360"]
    args += [
        "-vf",
        f"drawtext=text='Segue para mais!':expansion=none:fontfile={FONT}:"
        "fontsize=64:fontcolor=white:borderw=3:bordercolor=black:"
        "x=(w-tw)/2:y=(h-th)/2,fade=t=in:st=0:d=0.4",
        *NVENC, "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-shortest", out,
    ]
    run(args)


def probe(path):
    """Codec/rate/duration facts about one file."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[-1000:]}")
    data = json.loads(out.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    return {
        "video": {
            "fps": video.get("r_frame_rate"),
            "time_base": video.get("time_base"),
            "duration": float(video.get("duration", 0) or 0),
            "nb_frames": int(video.get("nb_frames", 0) or 0),
            "profile": video.get("profile"),
            "width": video.get("width"),
            "height": video.get("height"),
        },
        "audio": None if audio is None else {
            "sample_rate": audio.get("sample_rate"),
            "channels": audio.get("channels"),
        },
        "container_duration": float(data["format"].get("duration", 0) or 0),
    }


def last_frames_pts(path, count=40):
    """Presentation timestamps of the final `count` video frames."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    pts = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return [float(v) for v in pts[-count:]]


def keyframes(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return [float(line) for line in out.stdout.split() if line]


def assess(pieces, published, sum_expected, published_path):
    """Turn the measurements into pass/fail findings."""
    findings = []
    v = published["video"]
    end_card = pieces["end-card.mp4"]

    if end_card["video"]["fps"] != v["fps"]:
        findings.append(
            f"end card is {end_card['video']['fps']} fps but the published "
            f"track is {v['fps']} fps"
        )
    if end_card["video"]["time_base"] != v["time_base"]:
        findings.append(
            f"end card time_base {end_card['video']['time_base']} != "
            f"published {v['time_base']}"
        )

    # The end card's share of the output, measured from its first frame: the
    # last IDR keyframe that starts the end card, through the final frame.
    # This is the number that collapses to milliseconds when the bug is live.
    keys = keyframes(published_path)
    tail = last_frames_pts(published_path)
    if keys and tail:
        actual_end_card = max(tail) - keys[-1]
        expected_end_card = end_card["video"]["duration"]
        if abs(actual_end_card - expected_end_card) > 0.2:
            findings.append(
                f"end card occupies {actual_end_card:.3f}s in the published "
                f"file but is {expected_end_card:.3f}s long - the frames "
                "collapsed onto the main clip's last frames"
            )

    track_vs_container = abs(
        v["duration"] - published["container_duration"]
    )
    if track_vs_container > 0.5:
        findings.append(
            f"video track lasts {v['duration']:.3f}s but the container claims "
            f"{published['container_duration']:.3f}s"
        )

    rates = {name: p["audio"]["sample_rate"] for name, p in pieces.items()}
    if len(set(rates.values())) > 1:
        findings.append(
            f"audio sample rates differ across the pieces: {rates} - the "
            "concat silently adopts the first one and desyncs"
        )

    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true",
        help="keep the working directory and print its path",
    )
    parser.add_argument(
        "--unpinned-end-card", action="store_true",
        help="build the end card WITHOUT the -r fix, to reproduce the bug",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg and ffprobe must be on PATH", file=sys.stderr)
        return 2

    workdir = tempfile.mkdtemp(prefix="endcard-probe-")
    try:
        paths = {name: os.path.join(workdir, name) for name in
                 ("source.mp4", "cropped-teaser.mp4", "cropped-main.mp4",
                  "teaser.mp4", "main.mp4", "end-card.mp4", "published.mp4")}
        print(f"building pieces in {workdir} ...")
        build_source(paths["source.mp4"])
        build_cropped(paths["source.mp4"], paths["cropped-teaser.mp4"],
                      TEASER_SECONDS)
        build_cropped(paths["source.mp4"], paths["cropped-main.mp4"],
                      MAIN_SECONDS)
        burn(paths["cropped-teaser.mp4"], paths["teaser.mp4"], "teaser", "hook")
        burn(paths["cropped-main.mp4"], paths["main.mp4"], "main", "caption")
        build_end_card(paths["end-card.mp4"],
                       pin_rate=not args.unpinned_end_card)

        concat_list = os.path.join(workdir, "list.txt")
        with open(concat_list, "w", encoding="utf-8") as handle:
            for name in ("teaser.mp4", "main.mp4", "end-card.mp4"):
                handle.write(f"file '{paths[name]}'\n")
        run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "concat", "-safe", "0", "-i", concat_list,
             "-c", "copy", paths["published.mp4"]])

        pieces = {
            "teaser.mp4": probe(paths["teaser.mp4"]),
            "main.mp4": probe(paths["main.mp4"]),
            "end-card.mp4": probe(paths["end-card.mp4"]),
        }
        published = probe(paths["published.mp4"])

        sum_expected = sum(p["video"]["duration"] for p in pieces.values())
        sum_expected = max(sum_expected, published["container_duration"])

        report = {
            "pieces": pieces,
            "published": published,
            "sum_of_piece_durations": sum_expected,
            "duration_error": abs(
                published["video"]["duration"] - sum_expected
            ),
        }
        print(json.dumps(report, indent=2))

        findings = assess(pieces, published, sum_expected,
                          paths["published.mp4"])
        print()
        if findings:
            print("BOUNDARY FAILURES:")
            for finding in findings:
                print(f"  - {finding}")
        else:
            print("boundaries healthy: the end card survives the concat")
        if args.keep:
            print(f"\nworking directory kept at {workdir}")
        return 1 if findings else 0
    finally:
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
