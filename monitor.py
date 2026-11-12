import time
import os
import subprocess
import shutil
import threading
import queue
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
import argparse
import re

class NewVideoHandler(FileSystemEventHandler):
    def __init__(self, source_link=None, publish_interval_minutes=0, max_publish_retries=3,
                 dashboard_interval_seconds=0, selection_enabled=True, selection_limit=6,
                 selection_quiet_minutes=20, selection_batch_cap=0,
                 publish_queue_path=None, selection_state_path=None):
        self.processing_files = set()  # Track files currently being processed
        self.lock = threading.Lock()
        self.video_queue = queue.Queue()  # FIFO queue for video files
        self.queue_worker_thread = None
        self.stop_worker = threading.Event()
        self.source_link = source_link

        # Editing-side bookkeeping for the dashboard.
        self.edited_total = 0

        # Disk-backed publish queue + drip scheduler. Editing collects candidates;
        # a selection pass ranks the batch and only the best clips are queued, then
        # a background thread publishes them one at a time on the configured interval.
        from social_media_publisher.publish_queue import PublishQueue, PublishScheduler
        from social_media_publisher.clip_selection import (
            SelectionScheduler,
            SelectionStore,
            read_candidate,
        )
        root = Path(__file__).parent
        queue_path = Path(publish_queue_path) if publish_queue_path else root / "publish_queue.json"
        self.publish_queue = PublishQueue(str(queue_path))
        self.publish_scheduler = PublishScheduler(
            self.publish_queue,
            interval_minutes=publish_interval_minutes,
            max_retries=max_publish_retries,
            on_publish=self._on_publish_callback,
        )
        self.publish_interval_minutes = publish_interval_minutes

        # Batch selection: edited clips become candidates; once editing goes
        # quiet the batch is ranked and only the top `selection_limit` are queued.
        self.selection_enabled = selection_enabled
        selection_path = (
            Path(selection_state_path) if selection_state_path else root / "selection_state.json"
        )
        self.selection_store = SelectionStore(str(selection_path))
        self.selection_scheduler = SelectionScheduler(
            self.selection_store,
            self.publish_queue,
            limit=selection_limit,
            quiet_seconds=selection_quiet_minutes * 60.0,
            batch_cap=selection_batch_cap,
            enabled=selection_enabled,
            is_editing_busy=self.is_editing_busy,
        )
        # Candidates restored from a previous run have not been decided yet, but
        # their edits are long over, so the quiet period is treated as elapsed.
        if self.selection_store.pending():
            self.selection_scheduler.prime_as_quiet()
        self._reconcile_with_publish_queue()

        # Periodic dashboard ticker.
        self.dashboard_interval_seconds = dashboard_interval_seconds
        self.dashboard_thread = None
        self.stop_dashboard = threading.Event()

    def _reconcile_with_publish_queue(self):
        """Decide any restored candidate that a previous run already queued.

        Without this, a restart could rank a clip that is already waiting in the
        publish queue and enqueue it a second time.
        """
        queued = set(self.publish_queue.active_video_names())
        if not queued:
            return
        for candidate in self.selection_store.pending():
            if candidate.video_name in queued:
                self.selection_store.mark_decided(
                    candidate.video_name,
                    rank=0,
                    score=None,
                    selected=False,
                    reason="already in publish queue from a previous run",
                )

    def is_editing_busy(self):
        """True while a video is being edited, so the batch is left open."""
        return not self.video_queue.empty() or bool(self.processing_files)
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v")):
            file_path = Path(event.src_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New video detected: {file_path.name}")
            
            # Add to queue for processing
            self.video_queue.put(file_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Added {file_path.name} to processing queue (queue size: {self.video_queue.qsize()})")
    
    def start_queue_worker(self):
        """Start the queue worker thread"""
        if self.queue_worker_thread is None or not self.queue_worker_thread.is_alive():
            self.stop_worker.clear()
            self.queue_worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
            self.queue_worker_thread.start()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Queue worker thread started")
    
    def stop_queue_worker(self):
        """Stop the queue worker thread"""
        if self.queue_worker_thread and self.queue_worker_thread.is_alive():
            self.stop_worker.set()
            self.queue_worker_thread.join(timeout=5)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Queue worker thread stopped")

    # ---- publish scheduler + dashboard lifecycle ------------------------

    def start_publish_scheduler(self):
        """Start the publish scheduler (and optional dashboard ticker)."""
        self.publish_scheduler.start()
        mode = (f"{self.publish_interval_minutes} min interval"
                if self.publish_interval_minutes > 0 else "immediate")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Publish scheduler started ({mode})")
        if self.selection_enabled:
            self.selection_scheduler.start()
            limit = self.selection_scheduler.limit
            quiet = self.selection_scheduler.quiet_seconds / 60.0
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch selection started "
                  f"(top {limit} clips published after {quiet:g} min of quiet editing)")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Batch selection disabled — "
                  f"every edited clip is queued immediately")
        if self.dashboard_interval_seconds > 0:
            self.stop_dashboard.clear()
            self.dashboard_thread = threading.Thread(target=self._dashboard_loop, daemon=True)
            self.dashboard_thread.start()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Dashboard ticker started "
                  f"(every {self.dashboard_interval_seconds}s)")

    def stop_publish_scheduler(self):
        """Stop the publish scheduler and dashboard ticker."""
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.stop_dashboard.set()
            self.dashboard_thread.join(timeout=3)
        self.selection_scheduler.stop(timeout=5)
        self.publish_scheduler.stop(timeout=5)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Publish scheduler stopped")

    def _dashboard_loop(self):
        """Print the dashboard on an interval until stopped."""
        while not self.stop_dashboard.wait(self.dashboard_interval_seconds):
            print("\n" + self.render_dashboard() + "\n", flush=True)

    def editing_stats(self):
        """Snapshot of editing-side progress for the dashboard."""
        return {
            "pending": self.video_queue.qsize(),
            "in_progress": len(self.processing_files),
            "edited_total": self.edited_total,
        }

    def selection_stats(self):
        """Snapshot of batch-selection progress for the dashboard."""
        stats = self.selection_store.stats()
        last = self.selection_scheduler.last_edit_finished()
        return {
            "pending": stats["pending"],
            "decided": stats["decided"],
            "last_edit": (
                datetime.fromtimestamp(last).strftime("%H:%M:%S") if last else "n/a"
            ),
        }

    def render_dashboard(self):
        """Render a one-block status summary of editing + publishing."""
        e = self.editing_stats()
        p = self.publish_queue.stats()
        s = self.selection_stats()
        ts = datetime.now().strftime('%H:%M:%S')
        return (
            f"╭─ 📊 Auto-Clip Uploader Status — {ts} ─────╮\n"
            f"│ ✂️  Editing    pending: {e['pending']:<3} in-progress: {e['in_progress']:<3} done: {e['edited_total']}\n"
            f"│ 🏆 Selection  candidates: {s['pending']:<3} decided: {s['decided']:<3} last edit: {s['last_edit']}\n"
            f"│ 📤 Publishing pending: {p.get('pending', 0):<3} publishing: {p.get('publishing', 0):<3} "
            f"✅ published: {p.get('published', 0):<3} ❌ failed: {p.get('failed', 0)}\n"
            f"│ 📋 Queue total: {p.get('total', 0)}\n"
            f"╰───────────────────────────────────────────────╯"
        )

    def _on_publish_callback(self, entry, result):
        """One-line feedback after each publish attempt."""
        if result.get("success"):
            print(f"📤 Published: {entry.video_name}")
        else:
            print(f"⚠️  Publish of {entry.video_name} did not report success: "
                  f"{result.get('error', 'unknown')}")
    
    def _queue_worker(self):
        """Worker thread that processes videos from the queue in FIFO order"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Queue worker started, waiting for videos...")
        
        while not self.stop_worker.is_set():
            try:
                # Wait for a video with timeout to allow checking stop event
                file_path = self.video_queue.get(timeout=1.0)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing {file_path.name} from queue (remaining: {self.video_queue.qsize()})")
                
                # Process the file
                self._monitor_and_process_file(file_path)
                
                # Mark task as done
                self.video_queue.task_done()
                
            except queue.Empty:
                # No video in queue, continue checking
                continue
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in queue worker: {e}")
                continue
    
    def _monitor_and_process_file(self, file_path):
        """Monitor a file until it's fully copied, then process it"""
        if str(file_path) in self.processing_files:
            print(f"File {file_path.name} is already being processed, skipping")
            return
            
        self.processing_files.add(str(file_path))
        
        try:
            # Wait for file to be fully copied
            if self._wait_for_file_completion(file_path):
                self.process_video(file_path)
            else:
                print(f"File {file_path.name} was not completed properly, skipping")
        finally:
            self.processing_files.discard(str(file_path))
    
    def _wait_for_file_completion(self, file_path, max_wait=300):
        """Wait for a file to be fully copied by monitoring its size"""
        print(f"Waiting for {file_path.name} to be fully copied...")
        
        last_size = 0
        stable_count = 0
        start_time = time.time()
        check_interval = 2  # seconds
        
        while time.time() - start_time < max_wait:
            try:
                if not file_path.exists():
                    print(f"File {file_path.name} disappeared, aborting")
                    return False
                    
                current_size = file_path.stat().st_size
                
                if current_size == last_size and current_size > 0:
                    stable_count += 1
                    if stable_count >= 3:  # File size stable for 3 checks
                        print(f"File {file_path.name} appears to be fully copied ({current_size} bytes)")
                        return True
                else:
                    stable_count = 0
                    last_size = current_size
                    
                time.sleep(check_interval)
                
            except (OSError, IOError) as e:
                print(f"Error checking file {file_path.name}: {e}")
                time.sleep(check_interval)
                
        print(f"Timeout waiting for {file_path.name} to be fully copied")
        return False

    def process_video(self, video_path):
        """Process a video file using reelsfy"""
        with self.lock:
            video_name = Path(video_path).stem  # filename without extension
            # The watched folder is external input: a name containing path
            # separators or dot components must not be able to steer the
            # output directory outside output/.
            video_name = re.sub(r"[^A-Za-z0-9._-]+", "_", video_name).strip("._") or "untitled"
            output_base_dir = Path("output")
            video_output_dir = output_base_dir / video_name
            # Containment guard: the write sites below (logs, organized
            # outputs) must stay under output/ even if a future rename
            # weakens the name sanitization above.
            if video_output_dir.resolve().parent != output_base_dir.resolve():
                print(f"⚠️  Refusing unsafe output dir for {video_name!r}; skipping")
                return
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing video: {Path(video_path).name}")
            print(f"Output directory: {video_output_dir}")
            
            try:
                # Create output directory
                video_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Run reelsfy from within its directory
                reelsfy_dir = Path(__file__).parent / "reels-clips-automator"
                reelsfy_path = reelsfy_dir / "reelsfy.py"
                
                print("Starting reelsfy processing...")
                # Prepare base command
                command = ["python", str(reelsfy_path), "-f", str(video_path), "--output-dir", "../"+str(video_output_dir)]
                
                # Add video title if source link is present
                if self.source_link:
                    try:
                        print(f"🔗 Fetching title from: {self.source_link}")
                        # Prefer the yt-dlp sitting next to this interpreter (the conda
                        # env's up-to-date one) over a possibly stale one on PATH.
                        import sys as _sys
                        _env_ytdlp = os.path.join(os.path.dirname(_sys.executable), "yt-dlp")
                        yt_dlp_bin = _env_ytdlp if os.path.exists(_env_ytdlp) else "yt-dlp"
                        yt_command = [yt_dlp_bin, "--get-title", self.source_link]
                        
                        # Check if we are in a conda environment and need to use the same python
                        # But simpler is to assume yt-dlp is in the path or use subprocess directly
                        yt_result = subprocess.run(
                            yt_command,
                            capture_output=True,
                            text=True,
                            check=False
                        )
                        
                        if yt_result.returncode == 0:
                            # GreenBoost hooks print [gb_*] telemetry into the stdout of
                            # every Python process (yt-dlp included); strip those lines
                            # and keep the last remaining line, which is the title.
                            title_lines = [
                                l for l in yt_result.stdout.splitlines()
                                if l.strip() and not l.lstrip().startswith("[gb_")
                            ]
                            full_title = title_lines[-1].strip() if title_lines else ""
                            if not full_title:
                                raise ValueError("yt-dlp returned no title")
                            print(f"   Original Title: {full_title}")
                            
                            # Extract pattern: [ Show ][ ep #Number ]
                            # Regex: ^(\[\s*.+?\s*\])\s*(\[\s*ep\s*#?\d+\s*\])
                            match = re.search(r"^(\[\s*.+?\s*\])\s*(\[\s*ep\s*#?\d+\s*\])", full_title, re.IGNORECASE)
                            
                            if match:
                                video_title = f"{match.group(1)}{match.group(2)}"
                                print(f"   Formatted Title: {video_title}")
                            else:
                                # If pattern not found, use a shortened version or the full title? 
                                # Using full title might be too long for overlay.
                                # Let's try to be smart or just use it as is if it's short enough.
                                video_title = full_title
                                print(f"   Using full title as fallback")
                                
                            command.extend(["--video-title", video_title])
                        else:
                            print(f"⚠️  Failed to fetch title: {yt_result.stderr}")
                    except Exception as e:
                        print(f"⚠️  Error processing source link: {e}")
                    
                    # Also pass the source link to reelsfy for metadata
                    command.extend(["--source-link", self.source_link])
                
                print("Starting reelsfy processing...")
                result = subprocess.run(
                    command, 
                    cwd=str(reelsfy_dir),
                    capture_output=True,
                    text=True,
                    check=False
                )
                
                if result.returncode == 0:
                    print(f"✅ Successfully processed {Path(video_path).name}")
                    self._organize_output_files(reelsfy_dir, video_output_dir, Path(video_path).name)
                else:
                    print(f"❌ Error processing {Path(video_path).name}")
                    print(f"Error output: {result.stderr}")
                    
                    # Save error log
                    error_log_path = video_output_dir / "error.log"
                    with open(error_log_path, 'w') as f:
                        f.write(f"Processing failed at {datetime.now()}\n")
                        f.write(f"Return code: {result.returncode}\n")
                        f.write(f"Stdout: {result.stdout}\n")
                        f.write(f"Stderr: {result.stderr}\n")
                        
            except Exception as e:
                print(f"❌ Exception processing {Path(video_path).name}: {str(e)}")
                
                # Save exception log
                error_log_path = video_output_dir / "exception.log"
                with open(error_log_path, 'w') as f:
                    f.write(f"Exception occurred at {datetime.now()}\n")
                    f.write(f"Error: {str(e)}\n")
                    
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Finished processing {Path(video_path).name}")
    
    def _organize_output_files(self, reelsfy_dir, video_output_dir, video_name):
        """Move generated files to organized output directory"""
        try:
            # Look for generated files in reelsfy outputs directory
            # reelsfy_outputs_dir = reelsfy_dir / "outputs" / "input"
            reelsfy_outputs_dir = video_output_dir
            
            moved_files = []
            
            if reelsfy_outputs_dir.exists():
                import glob
                # ONLY accept final-* prefixed files (complete: audio + video + subtitles + overlays)
                final_videos = sorted(reelsfy_outputs_dir.glob("final-*.mp4"))
                if final_videos:
                    chosen = final_videos[0]
                    dest_path = video_output_dir / f"final_video_{chosen.stem}.mp4"
                    shutil.move(str(chosen), str(dest_path))
                    moved_files.append(dest_path.name)
                    print(f"Moved {chosen.name} to {dest_path.name}")
                else:
                    print(f"⚠️  No final video (final-*.mp4) found in {reelsfy_outputs_dir}")
                
                # Also move content/metadata files
                for specific_file in ["content.txt", "transcript.txt", "description.txt"]:
                    source_path = reelsfy_outputs_dir / specific_file
                    if source_path.exists():
                        dest_path = video_output_dir / specific_file
                        shutil.move(str(source_path), str(dest_path))
                        moved_files.append(specific_file)
                        print(f"Moved {specific_file}")
            
            if moved_files:
                print(f"Organized {len(moved_files)} files in {video_output_dir}")

                # Count a successful edit for the dashboard, then either collect the
                # clip as a selection candidate or queue it immediately.
                self.edited_total += 1
                if self.selection_enabled:
                    self._collect_candidate(video_output_dir, video_name)
                else:
                    self._publish_to_social_media(video_output_dir, video_name)
            else:
                print("No output files found to organize")

        except Exception as e:
            print(f"Error organizing output files: {e}")

    def _collect_candidate(self, video_output_dir, video_name):
        """Hold a finished clip back for the next ranking instead of queueing it."""
        try:
            from social_media_publisher.clip_selection import read_candidate

            candidate = read_candidate(
                str(video_output_dir), video_name, self.source_link
            )
            self.selection_store.add_candidate(candidate)
            # Restart the quiet period: the batch stays open while edits continue.
            self.selection_scheduler.on_edit_finished()
            stats = self.selection_store.stats()
            print(f"🏆 Candidate '{video_name}' held for selection "
                  f"({stats['pending']} waiting; best "
                  f"{self.selection_scheduler.limit} will be queued)")
        except Exception as e:
            # A clip we cannot even record must still reach the publish queue
            # rather than silently disappearing.
            print(f"⚠️  Could not record candidate '{video_name}': {e}")
            self._publish_to_social_media(video_output_dir, video_name)
    
    def _publish_to_social_media(self, video_output_dir, video_name):
        """Enqueue an edited video for drip publishing (non-blocking)."""
        try:
            entry = self.publish_queue.enqueue(str(video_output_dir), video_name, self.source_link)
            stats = self.publish_queue.stats()
            print(f"📥 Queued '{video_name}' for publishing "
                  f"(queue: {stats['pending']} pending, {stats['total']} total)")
        except Exception as e:
            print(f"📥 Failed to enqueue '{video_name}' for publishing: {e}")

def is_cifs_mount(path):
    """Check if a path is on a CIFS/SMB mount"""
    try:
        import subprocess
        result = subprocess.run(['stat', '-f', '-c', '%T', str(path)], 
                              capture_output=True, text=True, check=False)
        return 'cifs' in result.stdout.lower() or 'smb' in result.stdout.lower()
    except:
        return False

def build_arg_parser():
    """CLI surface. Every flag overrides the matching environment variable."""
    parser = argparse.ArgumentParser(description='Monitor folder for new video files to process and publish.')
    parser.add_argument('--link', help='Optional URL of the original video/source to include in description')
    parser.add_argument('--publish-delay', type=int, default=None,
                        help='Minutes between publishes (overrides PUBLISH_INTERVAL_MINUTES; 0=immediate)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable the periodic status dashboard')
    parser.add_argument('--no-selection', action='store_true',
                        help='Disable batch selection: queue every edited clip immediately')
    parser.add_argument('--select-limit', type=int, default=None,
                        help='How many of the best clips to publish per batch (overrides SELECTION_LIMIT)')
    parser.add_argument('--select-quiet-minutes', type=int, default=None,
                        help='Minutes of quiet editing before a batch is ranked (overrides SELECTION_QUIET_MINUTES)')
    parser.add_argument('--select-batch-cap', type=int, default=None,
                        help='Rank early once this many candidates are waiting (overrides SELECTION_BATCH_CAP; 0=no cap)')
    return parser

def main():
    # Load environment variables from the root .env file
    load_dotenv()

    video_folder = os.getenv("VIDEO_FOLDER")
    if not video_folder:
        print("Error: VIDEO_FOLDER not set in .env file.")
        exit(1)

    # Expand user path (e.g., ~) and make it absolute
    video_folder = os.path.expanduser(video_folder)
    video_folder_path = Path(video_folder)
    if not video_folder_path.is_absolute():
        # Make relative paths relative to the script location
        video_folder_path = Path(__file__).parent / video_folder

    output_folder = Path(__file__).parent / "output"

    # Check if the path is on a CIFS mount
    use_polling = is_cifs_mount(video_folder_path)
    observer_type = "PollingObserver (CIFS detected)" if use_polling else "Observer (native filesystem)"

    print("🎬 Video File Monitor Starting...")

    # Parse command line arguments
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.link:
        print(f"🔗 Source link enabled: {args.link}")

    # Resolve publishing cadence: CLI flag overrides env, which defaults to 0 (immediate).
    publish_interval = args.publish_delay if args.publish_delay is not None else int(
        os.getenv("PUBLISH_INTERVAL_MINUTES", "0"))
    max_retries = int(os.getenv("PUBLISH_MAX_RETRIES", "3"))
    dashboard_interval = 0 if args.no_dashboard else int(os.getenv("DASHBOARD_INTERVAL_SECONDS", "60"))

    # Resolve batch selection: CLI flag overrides env, which has in-code defaults.
    selection_enabled = (os.getenv("SELECTION_ENABLED", "true").strip().lower()
                         != "false") and not args.no_selection
    select_limit = args.select_limit if args.select_limit is not None else int(
        os.getenv("SELECTION_LIMIT", "6"))
    select_quiet = args.select_quiet_minutes if args.select_quiet_minutes is not None else int(
        os.getenv("SELECTION_QUIET_MINUTES", "20"))
    select_cap = args.select_batch_cap if args.select_batch_cap is not None else int(
        os.getenv("SELECTION_BATCH_CAP", "0"))
    select_limit = max(1, select_limit)

    if publish_interval > 0:
        print(f"📤 Drip publishing: one video every {publish_interval} minute(s)")
    else:
        print("📤 Publishing: immediate (each edited video published in order)")

    if selection_enabled:
        print(f"🏆 Batch selection: best {select_limit} clip(s) per batch, ranked after "
              f"{select_quiet} min of quiet editing"
              + (f" (early rank at {select_cap} candidates)" if select_cap > 0 else ""))
    else:
        print("🏆 Batch selection disabled — every edited clip is queued")

    print(f"👀 Monitoring: {video_folder_path.absolute()}")
    print(f"📁 Output to: {output_folder.absolute()}")
    print(f"🔍 Observer type: {observer_type}")
    print("🎯 Supported formats: .mp4, .mov, .avi, .mkv, .webm, .flv, .wmv, .m4v")

    # Create directories if they don't exist
    video_folder_path.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(exist_ok=True)

    event_handler = NewVideoHandler(
        source_link=args.link,
        publish_interval_minutes=publish_interval,
        max_publish_retries=max_retries,
        dashboard_interval_seconds=dashboard_interval,
        selection_enabled=selection_enabled,
        selection_limit=select_limit,
        selection_quiet_minutes=select_quiet,
        selection_batch_cap=select_cap,
    )

    # Start the queue worker and the publish scheduler
    event_handler.start_queue_worker()
    event_handler.start_publish_scheduler()

    observer = PollingObserver() if use_polling else Observer()
    observer.schedule(event_handler, str(video_folder_path), recursive=False)
    observer.start()

    print("✅ Monitor started. Waiting for video files...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
        observer.stop()
        event_handler.stop_queue_worker()
        event_handler.stop_publish_scheduler()
    observer.join()

    # Final dashboard summary.
    print("\n" + event_handler.render_dashboard())
    print("👋 Monitor stopped")

if __name__ == "__main__":
    main()
