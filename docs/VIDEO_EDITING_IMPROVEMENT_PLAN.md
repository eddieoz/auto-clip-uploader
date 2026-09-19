# Video Editing Improvement Plan — Social Media Appeal

> Systematic analysis of the auto-clip-uploader pipeline (`monitor.py` → `reelsfy.py` →
> `social_media_publisher/publish_queue.py` → Postiz) with a prioritized plan to make the
> generated reels more appealing on Instagram Reels / TikTok / YouTube Shorts.

---

## 1. Current pipeline (as found)

1. **Ingest** — `monitor.py` watches for new files; `reelsfy.py` copies input to `tmp/input_video.mp4`
   (or downloads via pytube).
2. **Transcribe** — `transcription_service.py` (whisper / faster-whisper) produces segment-level SRT
   (`reelsfy.py:1618`).
3. **Segment selection** — `TopicAnalyzer` picks a "concluding topic" segment (35–59 s),
   GPT writes title/description/tags (`reelsfy.py:1183`).
4. **Cut segment** — ffmpeg trims the segment (`reelsfy.py:353`).
5. **Reframe** — `generate_short` (`reelsfy.py:518`) crops to 1080×1920 with OpenCV Haar face
   detection every 5 s, random tracking/zoom modes, smoothing, optional RealESRGAN/GFPGAN.
6. **Burn overlays** — `generate_subtitle` (`reelsfy.py:2219`) **re-transcribes** the cropped clip,
   burns segment-level subtitles plus a date overlay and channel/title overlay.
7. **Publish** — queue + drip-publish via `social_media_publisher` to enabled Postiz channels.

### Strengths worth keeping
- Face-talking detection with look-ahead (`talking_face_detector.py`) already avoids static
  thumbnail faces — a real differentiator.
- Topic-completeness-based segment selection (vs. raw "viral" heuristics) produces coherent clips.
- 9:16 crop logic already guards aspect ratio (no stretching, `reelsfy.py:907–939`).
- Disk-backed drip queue prevents platform rate-limit flags.

---

## 2. Gap analysis — what makes these clips underperform

Ordered by impact on retention/appeal:

| # | Gap | Where | Why it hurts |
|---|-----|-------|--------------|
| G1 | Segment-level, bottom-aligned, small subtitles | `reelsfy.py:2390` | Modern Reels/TikTok use **large, centered, word-by-word karaoke captions**; this is the single biggest visual differentiator viewers expect. |
| G2 | No audio mastering | `reelsfy.py:985–990` | No loudness normalization (platforms normalize to ≈ −14 LUFS), no compression; clips sound quiet next to competitors. |
| G3 | No hook strategy | segment JSON / overlays | Nothing signals value in the first 1–3 s; retention drop happens immediately. No hook text overlay. |
| G4 | Silences and filler left in | whole pipeline | Long pauses kill retention; jump-cutting dead air is standard practice for talking-head reels. |
| G5 | Random editing choices | `reelsfy.py:546–551, 639–655` | Random tracking/zoom keeps visual variety (good intent), but the 9:16 crop **zooms onto the face and discards the screen content** — for screencast-style sources (camera in top third, screen content below) the actual payload is the screen, not the face. Zoom-in also resets every 5 s, creating "pumping". |
| G6 | Face centered vertically | `reelsfy.py:856–857` | Face is dead-center of the 1920-px frame; head should sit in the upper third with caption space below. |
| G7 | Overlays only half-identify the clip | `reelsfy.py:2394–2414` | Date + title overlays help viewers (and YouTube) identify the clip, but the date/title are **not** written into `description.txt`, so on platforms that re-crop or strip burned text the info is lost. |
| G8 | Double re-encode quality loss | `reelsfy.py:594–597, 989` | Intermediate `mp4v` (OpenCV) writer then NVENC re-encode; also `-ss` after `-i` decodes the whole source per segment (`reelsfy.py:380`). |
| G9 | No visual polish | overlays step | No color/saturation lift, no brand accent color, no end card, no CTA. |
| G10 | Center-crop fallback ignores audio | `reelsfy.py:947–967` | When no face is detected the crop centers the frame even if the speaker is off-frame. |
| G11 | No cold-open hook structure | segment JSON / timeline | Nothing signals value in the first 1–3 s, and the clip starts cold; a cut-away teaser showing the best moment before the full sequence is the standard Reels structure. |

---

## 3. Improvement plan

### Phase 1 — Highest impact, lowest risk (audio + captions)

**P1.1 Loudness normalization & gentle dynamics** *(targets G2)*
- Insert an audio filter chain in the final mux (`generate_subtitle` ffmpeg command):
  `loudnorm=I=-14:LRA=11:tp=-1.5,acompressor=threshold=0.05:ratio=3:makeup=1.2`
- Run two-pass loudnorm when processing time allows; single-pass otherwise.
- Acceptance: `ffmpeg loudnorm print` report ≈ −14 LUFS integrated; dialog noticeably level-matched
  to popular reels when compared side by side.

**P1.2 Word-level "karaoke" captions** *(targets G1, G7)*
- Extend `transcription_service.py` transcribers to return **word timestamps**
  (faster-whisper: `word_timestamps=True`; whisper: same flag).
- Generate an **ASS subtitle file** instead of SRT for burning:
  - Font: Montserrat/Poppins **ExtraBold**, uppercase, 54–64 px at 1080×1920.
  - Placement: bottom-center **above the safe zone** (text block centered around y≈1450,
    never below y≈1600 where Reels UI sits).
  - 2–3 words per group, **current word highlighted** in a brand accent color
    (`\c&H...&`) or with a per-word pop scale (`\fscx/\fscy` animation).
  - Set `PlayResX=1080, PlayResY=1920` explicitly so `Fontsize` is predictable
    (the current `Fontsize=6` at `reelsfy.py:2390` only works by accident of ASS defaults).
- Reuse the master transcript (offset by segment start) instead of re-transcribing the cropped
  clip; keep the re-transcribe path only as fallback.
- Acceptance: captions readable at arm's length on a phone, word highlight tracks speech within
  ±100 ms, no text in the bottom 250 px / top 120 px UI zones.

**P1.3 Silence removal / jump cuts** *(targets G4)*
- Using word timestamps from P1.2, build an edit list cutting intra-segment pauses > 0.6 s and
  leading/trailing silence, with 120 ms padding per cut.
- Implement as a concat of trimmed pieces (ffmpeg `select`/`aselect` or segment re-cut from the
  **original** source so the OpenCV crop stage stays frame-accurate to the new timeline).
- Acceptance: no dead air longer than ~0.7 s in the final clip; speech never clipped mid-word
  (verify on 3 sample videos).

### Phase 2 — Framing, hook structure & overlays

**Context from review:** source videos are mostly **screencast-style** — camera in the top third,
screen content below. The existing random tracking/zoom modes were added on purpose to change the
scene and keep viewers engaged, so variety stays; what must change is that the 9:16 face crop
throws away the screen content, which is the actual payload of the video.

**P2.1 Layout-aware reframing (screencast-first)** *(targets G5, G6)*
- Detect the layout instead of assuming a talking head:
  - If the face occupies a small top-third region and the rest of the frame has screen-like
    content (high edge density / large UI rectangles) → **screencast layout**.
  - Only fall back to the current face-following crop when there is no meaningful screen content.
- In screencast layout, keep the screen as the hero and preserve visual variety (keeping the
  spirit of the current random modes) with **content-aware variation**:
  - **PiP style (default):** scale the screen capture to fill the 1080×1920 frame (cropping the
    letterbox bars), then overlay the camera as a **circular face-cam badge** in the top corner,
    tracking the face with the existing smoothing. This is the dominant format for coding/demo
    reels.
  - **Screen-zoom style:** follow the screen action (cursor/motion energy) with a 9:16 crop over
    the screen region — creates the "scene change" the random zoom was simulating, but it moves
    to where the content actually is.
  - Keep occasional wide face close-ups (the current behavior) for b-roll-like rhythm, but
    schedule them (e.g., alternate on a fixed pattern, or pick the mode from GPT segment
    metadata) instead of per-5 s dice rolls.
- Face-cam and any zoom still use the existing EMA smoothing (`reelsfy.py:746–767`) on **every**
  frame, then an interpolated crop path (one-euro or Savitzky-Golay) to remove micro-jitter.
- Keep the slow continuous push-in idea but apply it over the whole clip in PiP mode only
  (screen scales 100% → 96%), so it adds life without resetting every 5 s.

**P2.2 Cold-open hook teaser** *(targets G3, G11)*
- Add a `hook` field to the segment JSON produced by `generate_viral` (GPT picks the single most
  engaging sentence, 2–5 s, and a 4–7 word hook text in `{title_language}`).
- Build the final clip as **teaser → full sequence**:
  1. Cut the teaser directly from the source segment at the hook sentence's timestamps
     (cold open showing the best moment).
  2. Burn the hook text large over the teaser (same caption style as P1.2) for its duration.
  3. Hard-cut into the main clip from the beginning and play it in sequence; optionally fade the
     hook text out over the first ~0.5 s of the main clip.
- Reuse `workspace/backgrounds/reels` + the thumbnail generator (`generate_thumbnail`,
  `reelsfy.py:407`) as the fallback teaser when the hook sentence is too short or unusable —
  the hook card (background + hook text) plays for ~2 s instead of a cold-open cut.
- Acceptance: first frame of the file is the teaser; hook text legible at t=1 s; teaser audio is
  the actual hook sentence (not silence); main clip plays uncut after the cut.

**P2.3 Overlays: keep, but make them survivable** *(targets G7)*
- **Keep the date and title/channel overlays** — they identify the clip for viewers and on
  YouTube — but restyle: smaller (≈32 px), 60–70% opacity, moved fully inside the safe zone
  (top overlay at y≥120 px; title bottom-right at least 250 px above the bottom edge).
- Additionally write the same information into the description so it survives platforms that
  re-crop or strip burned text: append `Date: DD/MM/YYYY` and the overlay title to
  `description.txt` in `save_metadata` (`reelsfy.py:1711`). Rule of thumb going forward:
  **anything burned in the video must also be in the description.**
- Keep reusing the master transcript (offset by segment start) for subtitles instead of
  re-transcribing the cropped clip (see P1.2).

**P2.4 Faster, cleaner encode chain** *(targets G8)*
- `generate_segments`: move `-ss {start} -to {end}` **before** `-i` (fast seek) —
  `reelsfy.py:380` currently decodes the entire source per segment.
- Replace the OpenCV `mp4v` intermediate with a lossless-ish intermediate
  (`-c:v h264_nvenc -qp 0` or `libx264 -crf 10`) or better: compute the crop path in OpenCV,
  then do the final crop/scale/encode in a single ffmpeg pass using `crop=` expressions fed by
  the tracking data — one encode instead of two.
- Normalize fps to 30 on output (`fps=30`) for consistent platform behavior.

### Phase 3 — Polish & differentiators

**P3.1 Visual grade** *(targets G9)*
- Mild grade in the final pass: `eq=saturation=1.06:contrast=1.02,unsharp=5:5:0.6`.
- Optional per-channel LUT file in `.env` (`VIDEO_LUT=path`).

**P3.2 End card / CTA** *(targets G3, G9)*
- Last 1.5 s: fade to brand-colored end card with a CTA drawn from the metadata
  ("Follow for more") — ffmpeg `fade=out` + `drawtext`.

**P3.3 Music bed (optional, flag-gated)** *(targets G9)*
- `amix` a royalty-free loop at ≈ −24 dB under speech; library of loops in `workspace/music/`.
  Off by default. No ducking/sidechain needed — the source videos are already recorded with
  ducked music, so a plain low-level mix is enough.

**P3.4 Audio-aware fallback crop** *(targets G10)*
- In center-crop fallback, pan toward the active speaker using existing audio energy analysis
  (`enhanced_audio_analyzer`) or keep last known face position instead of hard centering.

**P3.5 Thumbnail upgrade**
- Current smile-detection thumbnail (`find_smile`, `generate_thumbnail`) is okay; add the hook
  text instead of the raw title and pick the highest face-quality frame near a sentence start.

---

## 4. Quick bug fixes found during analysis (do alongside Phase 1)

1. `generate_short` returns `None` on the success path, so `direct_process_file` always logs
   "Error: Failed to create reel" (`reelsfy.py:1754–1761`). Return `True`/output path.
2. `x, y, w, h` leak from the 5-second interval branch — if frame-by-frame tracking fails the
   crop silently uses the last interval's box (`reelsfy.py:724–738`). Reset/guard stale state.
3. `import whisper` at module top is unused and slow (module import loads torch early).
4. pytube is unmaintained and already broken with YouTube changes — swap to yt-dlp
   (requirements already pin yt-dlp ≥ 2026.6.9 elsewhere in the repo).
5. `cv2.waitKey(1)` in the frame loop (`reelsfy.py:974`) is meaningless in headless runs.
6. Random mode prints run even with `debug_faces=False` (`reelsfy.py:655`) — log noise.

---

## 5. Suggested execution order

1. P1.1 + P1.2 (biggest visible/audible win) → 2. P1.3 jump cuts → 3. P2.3 overlay restyle +
   description fallback → 4. P2.2 cold-open hook teaser → 5. P2.1 layout-aware reframing
   (screencast-first) → 6. P2.4 encode chain → 7. Phase 3 items as capacity allows.

Note: P1.3 (jump cuts) must be built on the teaser + main-sequence timeline from P2.2 — the
edit list is computed per piece (teaser and main clip), not across the cut point.

Each item should land with a test in `tests/` (per AGENTS.md), e.g.:
- test that the ASS builder groups words and highlights the spoken word at timestamp T;
- test that the silence edit-list never removes audio overlapping speech words;
- test that generated filter strings contain loudnorm and safe-zone-compliant positions.
