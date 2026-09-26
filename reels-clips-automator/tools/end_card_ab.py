#!/usr/bin/env python3
"""Story C4: A/B the published reel with and without the end card.

Runs both arms over a REAL input video using reelsfy's own assembly
function (`assemble_hook_teaser_final`) and its own end-card builder, then
probes both outputs and prints a comparison. Arm A is the end card ON,
arm B is the shipped default (OFF).

    python3 reels-clips-automator/tools/end_card_ab.py "input/foo.mp4" --out /tmp/c4

This is the local half of C4. It cannot tell you whether Instagram's
transcoder accepts the result, nor whether the CTA helps retention — only
an upload answers that. What it does answer, on real content rather than
synthetic fixtures, is whether both arms assemble into a single
well-formed video stream with the end card intact.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import MagicMock

HERE = os.path.dirname(os.path.abspath(__file__))
REELSFY_DIR = os.path.abspath(os.path.join(HERE, os.pardir))


def load_reelsfy():
    """Import reelsfy with the heavy optional deps stubbed.

    reelsfy imports librosa/numba at module scope, which is not importable
    in every environment (and is irrelevant to assembling a video). The
    video path we exercise below does not touch any of it.
    """
    sys.path.insert(0, REELSFY_DIR)
    for name in ("pytube", "whisper", "audio_analyzer",
                 "enhanced_audio_analyzer", "librosa", "numba"):
        try:
            __import__(name)
        except Exception:
            sys.modules[name] = MagicMock()
    sys.modules["transcription_service"] = MagicMock()
    import reelsfy
    return reelsfy


REELSFY = load_reelsfy()


def run(args, what):
    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"{what} failed (rc={result.returncode}): {' '.join(args)}\n"
            f"{result.stderr[-2000:]}"
        )
    return result


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr[-800:]}")
    data = json.loads(out.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    keys = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-skip_frame", "nokey", "-show_entries", "frame=pts_time",
         "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    return {
        "fps": video.get("r_frame_rate"),
        "time_base": video.get("time_base"),
        "video_duration": round(float(video.get("duration", 0) or 0), 3),
        "container_duration": round(
            float(data["format"].get("duration", 0) or 0), 3
        ),
        "nb_frames": int(video.get("nb_frames", 0) or 0),
        "width": video.get("width"),
        "height": video.get("height"),
        "profile": video.get("profile"),
        "audio_rate": (audio or {}).get("sample_rate"),
        "audio_channels": (audio or {}).get("channels"),
        "keyframes": [round(float(k), 3) for k in keys.stdout.split()],
    }


def build_main_piece(source, out, seconds):
    """The main clip: 9:16, overlay burn, audio mastering - the same shape
    generate_short + generate_subtitle produce, using reelsfy's own
    assembly constants so the piece is byte-compatible with the end card.
    """
    fps = REELSFY.ASSEMBLY_FPS
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", source, "-t", str(seconds),
        "-vf",
        f"crop=608:1080:656:0,scale=1080:1920,setsar=1,"
        f"drawtext=text='c4 arm':expansion=none:"
        f"fontfile={REELSFY.OVERLAY_FONTFILE}:fontsize=64:fontcolor=white:"
        f"borderw=3:bordercolor=black:x=(w-tw)/2:y=100",
        "-af", REELSFY.AUDIO_MASTERING_FILTER,
        "-c:v", "h264_nvenc", "-profile:v", "high", "-rc:v", "vbr",
        "-qp", "18", "-b:v", "10000k", "-maxrate:v", "12000k",
        "-bufsize:v", "15000k", "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-video_track_timescale", str(REELSFY.ASSEMBLY_VIDEO_TIMESCALE),
        "-c:a", "aac", "-b:a", "192k",
        "-ar", str(REELSFY.ASSEMBLY_AUDIO_RATE),
        "-ac", str(REELSFY.ASSEMBLY_AUDIO_CHANNELS),
        "-shortest", out,
    ], "main piece")


def build_end_card(out):
    """The end card, via reelsfy's own builder and its own generate()."""
    cwd = os.getcwd()
    os.chdir(REELSFY_DIR)
    try:
        if not os.path.isdir("tmp"):
            os.makedirs("tmp", exist_ok=True)
        print("  end card command: " + REELSFY.build_end_card_command()[:110]
              + " ...")
        path = REELSFY.generate_end_card()
        # generate_end_card() returns a path relative to reelsfy.py's own
        # directory; resolve it before leaving that directory.
        resolved = os.path.abspath(path) if path else None
    finally:
        os.chdir(cwd)
    if not resolved:
        raise RuntimeError("generate_end_card() returned None")
    shutil.copy(resolved, out)
    return out


def assemble(teaser, main, end_card, published, workdir):
    """Assemble with reelsfy's real ladder."""
    cwd = os.getcwd()
    os.chdir(REELSFY_DIR)
    try:
        if not os.path.isdir("tmp"):
            os.makedirs("tmp", exist_ok=True)
        ok = REELSFY.assemble_hook_teaser_final(
            teaser, main, os.path.abspath(published),
            concat_list_path=os.path.join(workdir, "concat.txt"),
            end_card_path=end_card,
        )
    finally:
        os.chdir(cwd)
    if not ok:
        raise RuntimeError("assemble_hook_teaser_final() returned False")
    return published


def run_arm(label, source, outdir, seconds, workdir):
    """Build and assemble one arm; returns its probe dict and path."""
    print(f"\n=== arm {label} ===")
    main = os.path.join(workdir, f"main-{label}.mp4")
    build_main_piece(source, main, seconds)
    print("  main piece: " + json.dumps(probe(main)))

    end_card = None
    if label == "A (end card ON)":
        end_card = build_end_card(os.path.join(workdir, "end-card.mp4"))
        print("  end card:  " + json.dumps(probe(end_card)))

    published = os.path.join(outdir, f"c4-arm-{label[0]}.mp4")
    assemble(None, main, end_card, published, workdir)
    result = probe(published)
    print("  published: " + json.dumps({k: v for k, v in result.items()
                                         if k != "keyframes"}))
    return result, published, probe(main), (probe(end_card) if end_card else None)


def compare(arm_a, main_a, card_a, arm_b, main_b):
    """The C4 question: does the end card survive, and does it break the file?"""
    findings = []
    if card_a is None:
        return ["arm A produced no end card - the A/B did not happen"]
    for name, piece in (("A", arm_a), ("B", arm_b)):
        if piece["fps"] != main_b["fps"] and name == "A":
            findings.append(
                f"arm A is {piece['fps']} fps but the main piece is "
                f"{main_b['fps']} fps"
            )
    if arm_a["time_base"] != arm_b["time_base"]:
        findings.append(
            f"arm A time_base {arm_a['time_base']} != arm B "
            f"{arm_b['time_base']} - the end card is not byte-compatible"
        )
    if abs(arm_a["video_duration"] - arm_b["video_duration"]) < 0.5:
        findings.append(
            f"arm A is only {arm_a['video_duration'] - arm_b['video_duration']:+.3f}s "
            f"longer than arm B; the 1.5s end card did not survive"
        )
    if abs(arm_a["video_duration"] - arm_a["container_duration"]) > 0.5:
        findings.append(
            f"arm A video track lasts {arm_a['video_duration']}s but the "
            f"container claims {arm_a['container_duration']}s"
        )
    for name, piece in (("A", arm_a), ("B", arm_b)):
        if piece["audio_rate"] != main_b["audio_rate"]:
            findings.append(
                f"arm {name} audio is {piece['audio_rate']} but the main "
                f"piece is {main_b['audio_rate']} - silent desync risk"
            )
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="a real input video")
    parser.add_argument("--out", default="/tmp/c4-ab", help="output dir")
    parser.add_argument(
        "--seconds", type=float, default=40.0,
        help="how much of the source to use (the clip cap is 59s)",
    )
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        print("ffmpeg and ffprobe must be on PATH", file=sys.stderr)
        return 2
    if not os.path.exists(args.source):
        print(f"no such file: {args.source}", file=sys.stderr)
        return 2

    os.makedirs(args.out, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="c4-ab-")
    try:
        print(f"source: {args.source}")
        print("  " + json.dumps(
            {k: v for k, v in probe(args.source).items() if k != "keyframes"}
        ))
        arm_a, path_a, main_probe, card_probe = run_arm(
            "A (end card ON)", args.source, args.out, args.seconds, workdir
        )
        arm_b, path_b, _, _ = run_arm(
            "B (end card OFF)", args.source, args.out, args.seconds, workdir
        )

        findings = compare(arm_a, main_probe, card_probe, arm_b, main_probe)

        report = {
            "source": args.source,
            "arm_A_end_card_on": arm_a,
            "arm_B_end_card_off": arm_b,
            "delta_seconds": round(
                arm_a["video_duration"] - arm_b["video_duration"], 3
            ),
            "findings": findings,
            "files": {"A": path_a, "B": path_b},
        }
        with open(os.path.join(args.out, "c4-ab-report.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)

        print("\n" + "=" * 66)
        print(f"arm A (end card ON) : {arm_a['video_duration']}s  "
              f"{arm_a['fps']}  {arm_a['time_base']}  "
              f"audio {arm_a['audio_rate']}  -> {path_a}")
        print(f"arm B (end card OFF): {arm_b['video_duration']}s  "
              f"{arm_b['fps']}  {arm_b['time_base']}  "
              f"audio {arm_b['audio_rate']}  -> {path_b}")
        print(f"delta: {report['delta_seconds']}s")
        print("=" * 66)
        if findings:
            print("FINDINGS:")
            for finding in findings:
                print(f"  - {finding}")
        else:
            print("both arms are well-formed single streams; the end card "
                  "survives the assembly")
        print(f"\nreport: {os.path.join(args.out, 'c4-ab-report.json')}")
        return 1 if findings else 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
