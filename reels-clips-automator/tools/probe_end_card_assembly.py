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
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REELSFY = os.path.join(HERE, os.pardir, "reelsfy.py")

# The settings under test. They are READ OUT OF reelsfy.py rather than
# repeated here, so the probe cannot drift away from the pipeline it checks —
# but they are read as text, not by importing reelsfy, which would drag in
# the whole transcription stack (librosa/numba) for four integers.
def pipeline_constants():
    """Pull the assembly constants out of reelsfy.py's source.

    Parsed with `ast`, not regex. A `^NAME = ...` regex with re.MULTILINE
    also matches text INSIDE a docstring or comment block sitting at column
    0, so a changelog note quoting "ASSEMBLY_FPS = 25" would silently
    poison the probe with a wrong-but-plausible value — and the probe would
    still exit 0, which is the worst way for it to fail. `ast.parse` only
    sees real assignments, so that class of shadowing cannot happen.
    """
    source = open(REELSFY, encoding="utf-8").read()
    tree = ast.parse(source, filename=REELSFY)

    # Only module-level assignments. A constant redefined inside a function
    # or class is not the pipeline default we are asserting against.
    assigned = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned[target.id] = node.value

    def grab(name):
        if name not in assigned:
            raise SystemExit(
                f"{name} not found as a module-level assignment in "
                f"{REELSFY}; the probe needs updating"
            )
        try:
            return ast.literal_eval(assigned[name])
        except ValueError:
            raise SystemExit(
                f"{name} in {REELSFY} is no longer a plain literal "
                f"(line {assigned[name].lineno}); the probe needs updating"
            )

    def grab_wrapped(name):
        """A string constant the author wrapped across lines.

        `ast.literal_eval` handles the parenthesised form directly; this
        exists only to name the failure usefully.
        """
        if name not in assigned:
            raise SystemExit(
                f"{name} not found as a module-level assignment in "
                f"{REELSFY}; the probe needs updating"
            )
        return grab(name)

    return {
        "fps": grab("ASSEMBLY_FPS"),
        "timescale": grab("ASSEMBLY_VIDEO_TIMESCALE"),
        "audio_rate": grab("ASSEMBLY_AUDIO_RATE"),
        "audio_channels": grab("ASSEMBLY_AUDIO_CHANNELS"),
        "mastering": grab_wrapped("AUDIO_MASTERING_FILTER"),
        "font": grab_wrapped("OVERLAY_FONTFILE"),
    }


CONST = pipeline_constants()
# Bound to locals: this module must import on Python 3.9, where an
# f-string cannot nest CONST["x"] inside its own double quotes.
FPS = CONST["fps"]
TIMESCALE = CONST["timescale"]
AUDIO_RATE = CONST["audio_rate"]
AUDIO_CHANNELS = CONST["audio_channels"]
MASTERING = CONST["mastering"]
FONT = CONST["font"]
NVENC = [
    "-c:v", "h264_nvenc", "-profile:v", "high", "-rc:v", "vbr",
    "-qp", "18", "-b:v", "10000k", "-maxrate:v", "12000k",
    "-bufsize:v", "15000k", "-pix_fmt", "yuv420p",
]
# Every piece the pipeline encodes pins the timescale, so the probe must too:
# otherwise changing ASSEMBLY_VIDEO_TIMESCALE in reelsfy.py would leave this
# tool reporting "healthy" while the real assembly broke.
TIMESCALE_ARGS = ["-video_track_timescale", str(TIMESCALE)]
TEASER_SECONDS = 4
MAIN_SECONDS = 40


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
        "-f", "lavfi", "-i", f"sine=frequency=300:r={AUDIO_RATE}:d=90",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", str(AUDIO_RATE),
        "-ac", str(AUDIO_CHANNELS), "-shortest", path,
    ])


def build_cropped(source, out, seconds):
    """The 9:16 crop, the way generate_short does it (note the pinned -r)."""
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", source, "-t", str(seconds),
        "-vf", "crop=608:1080:656:0,scale=1080:1920,setsar=1",
        "-r", str(FPS), "-pix_fmt", "yuv420p", *NVENC, *TIMESCALE_ARGS,
        "-c:a", "aac", "-b:a", "192k",
        "-ar", str(AUDIO_RATE),
        "-ac", str(AUDIO_CHANNELS), "-shortest", out,
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
        "-af", MASTERING,
        *NVENC, "-r", str(FPS), *TIMESCALE_ARGS,
        "-c:a", "aac", "-b:a", "192k",
        "-ar", str(AUDIO_RATE),
        "-ac", str(AUDIO_CHANNELS), out,
    ])


def build_end_card(out, pin_rate=True):
    """The end card, the way build_end_card_command() builds it.

    `pin_rate=False` reproduces the PRE-C3 state, which is what
    --unpinned-end-card is for: build_end_card_command() used to omit -r, so
    the lavfi color source defaulted to 25 fps with a 1/12800 timebase while
    every other piece is 30 fps / 1/15360 (the values the pipeline pins in
    reelsfy.ASSEMBLY_FPS / ASSEMBLY_VIDEO_TIMESCALE). C3 fixed that in the
    real builder; this flag keeps the broken shape available to prove the
    probe still catches it.

    Note this is a hand-rolled copy of the command, not a call into
    reelsfy.py — the constants are read live from its source, but the
    command shape is not. The test suite is what guards the real builder
    (see tests/test_end_card_assembly.py); use end_card_ab.py when you want
    the actual reelsfy functions driven end to end.
    """
    args = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-hwaccel", "cuda",
        "-f", "lavfi", "-i", "color=c=0x14213D:s=1080x1920:d=1.5",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
    ]
    if pin_rate:
        args += ["-r", str(FPS), *TIMESCALE_ARGS]
    args += [
        "-vf",
        f"drawtext=text='Segue para mais!':expansion=none:fontfile={FONT}:"
        "fontsize=64:fontcolor=white:borderw=3:bordercolor=black:"
        "x=(w-tw)/2:y=(h-th)/2,fade=t=in:st=0:d=0.4",
        *NVENC, "-c:a", "aac", "-b:a", "192k",
        "-ar", str(AUDIO_RATE),
        "-ac", str(AUDIO_CHANNELS), "-shortest", out,
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


def all_frames_pts(path):
    """Presentation timestamps of every video frame, in decode order.

    Used to locate the end card by frame count rather than by keyframe:
    see the occupancy check in assess().
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    pts = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return [float(v) for v in pts if v != "N/A"]


def all_frames_pts(path):
    """Presentation timestamps of every video frame, in decode order.

    Used to locate the end card by frame count rather than by keyframe:
    see the occupancy check in assess().
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    pts = [line.strip() for line in out.stdout.splitlines() if line.strip()]
    return [float(v) for v in pts if v != "N/A"]


def keyframes(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
         "-show_entries", "frame=pts_time", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return [float(line) for line in out.stdout.split() if line]


def assess(pieces, published, raw_sum, published_path):
    """Turn the measurements into pass/fail findings.

    `raw_sum` is the plain sum of the three piece durations, NOT a max()ed
    value: the gap between that sum and the published container is exactly
    the silent desync signal, and taking a max() over it would collapse a
    0.121s discrepancy down to noise before anything looks at it.
    """
    findings = []
    v = published["video"]
    end_card = pieces["end-card.mp4"]
    expected_end_card = end_card["video"]["duration"]

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

    # The end card's share of the output, located by FRAME COUNT rather than
    # by keyframes. The earlier version took the last IDR as the boundary,
    # which only works while the end card holds exactly one — a healthy card
    # with a tuned GOP was reported as collapsed. Guessing a keyframe
    # "near" the expected start is no better: in the collapsed case every
    # end-card frame is on top of the main clip's last frames, so any
    # deadline heuristic reports a meaningless span instead of the
    # 100 ms the end card really occupies.
    #
    # The end card is the last `end_card_frames` frames of the published
    # file, so the boundary is frame [total - end_card_frames] and the
    # occupancy is last_pts - that_pts. Exact, and immune to GOP structure.
    pts = all_frames_pts(published_path)
    end_card_frames = end_card["video"].get("nb_frames")
    if pts and end_card_frames:
        boundary_index = len(pts) - int(end_card_frames) - 1
        if 0 <= boundary_index < len(pts) - 1:
            actual_end_card = pts[-1] - pts[boundary_index]
            if actual_end_card < 0:
                # Not a formatting quirk: a negative span means the end
                # card's frames do not come after the main clip's last frame
                # at all — they sit on top of it. That inversion IS the
                # collapse, so say so rather than printing a negative length.
                findings.append(
                    f"end card frames are interleaved with the main clip's "
                    f"last frames (span {actual_end_card:.3f}s where "
                    f"{expected_end_card:.3f}s was expected) - the frames "
                    "collapsed onto the main clip's last frames"
                )
            elif abs(actual_end_card - expected_end_card) > 0.2:
                findings.append(
                    f"end card occupies {actual_end_card:.3f}s in the "
                    f"published file but is {expected_end_card:.3f}s long - "
                    "the frames collapsed onto the main clip's last frames"
                )

    track_vs_container = abs(
        v["duration"] - published["container_duration"]
    )
    if track_vs_container > 0.5:
        findings.append(
            f"video track lasts {v['duration']:.3f}s but the container claims "
            f"{published['container_duration']:.3f}s"
        )

    # The sum-vs-container gap: the signature of concat adopting one piece's
    # audio parameters and desyncing the rest. This compares the published
    # file against the pieces, so it can actually fire.
    desync = published["container_duration"] - raw_sum
    if abs(desync) > 0.2:
        findings.append(
            f"published container is {desync:+.3f}s off the sum of the "
            f"pieces ({raw_sum:.3f}s) - the concat adopted mismatched "
            "stream parameters and desynced"
        )

    # Audio shape: the published file must match the main clip it was built
    # from. Comparing the probe's own three pieces to each other was
    # vacuous - it builds them all with the same constant and could never
    # disagree with itself.
    main_audio = pieces["main.mp4"]["audio"]
    pub_audio = published["audio"]
    if pub_audio["sample_rate"] != main_audio["sample_rate"]:
        findings.append(
            f"published audio is {pub_audio['sample_rate']} but the main "
            f"piece is {main_audio['sample_rate']} - silent desync risk"
        )
    if pub_audio["channels"] != main_audio["channels"]:
        findings.append(
            f"published audio has {pub_audio['channels']} channels but the "
            f"main piece has {main_audio['channels']}"
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

        # The PLAIN sum. Taking a max() with the container duration here
        # silently hid the one number worth watching: how far the published
        # file is from the sum of what went into it. Both are reported.
        raw_sum = sum(p["video"]["duration"] for p in pieces.values())
        track_sum_delta = published["video"]["duration"] - raw_sum
        container_sum_delta = published["container_duration"] - raw_sum

        report = {
            "pieces": pieces,
            "published": published,
            "sum_of_piece_durations": raw_sum,
            "video_track_minus_sum": round(track_sum_delta, 3),
            "container_minus_sum": round(container_sum_delta, 3),
        }
        print(json.dumps(report, indent=2))

        findings = assess(pieces, published, raw_sum,
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
