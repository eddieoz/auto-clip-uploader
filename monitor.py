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
                 dashboard_interval_seconds=0):
        self.processing_files = set()  # Track files currently being processed
        self.lock = threading.Lock()
        self.video_queue = queue.Queue()  # FIFO queue for video files
        self.queue_worker_thread = None
        self.stop_worker = threading.Event()
        self.source_link = source_link

        # Editing-side bookkeeping for the dashboard.
        self.edited_total = 0

        # Disk-backed publish queue + drip scheduler. Editing appends to the queue;
        # a background thread publishes one video at a time on the configured interval.
        from social_media_publisher.publish_queue import PublishQueue, PublishScheduler
        queue_path = Path(__file__).parent / "publish_queue.json"
        self.publish_queue = PublishQueue(str(queue_path))
        self.publish_scheduler = PublishScheduler(
            self.publish_queue,
            interval_minutes=publish_interval_minutes,
            max_retries=max_publish_retries,
            on_publish=self._on_publish_callback,
        )
        self.publish_interval_minutes = publish_interval_minutes

        # Periodic dashboard ticker.
        self.dashboard_interval_seconds = dashboard_interval_seconds
        self.dashboard_thread = None
        self.stop_dashboard = threading.Event()
        
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

    def render_dashboard(self):
        """Render a one-block status summary of editing + publishing."""
        e = self.editing_stats()
        p = self.publish_queue.stats()
        ts = datetime.now().strftime('%H:%M:%S')
        return (
            f"╭─ 📊 Auto-Clip Uploader Status — {ts} ─────╮\n"
            f"│ ✂️  Editing    pending: {e['pending']:<3} in-progress: {e['in_progress']:<3} done: {e['edited_total']}\n"
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
            output_base_dir = Path("output")
            video_output_dir = output_base_dir / video_name
            
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

                # Count a successful edit for the dashboard, then enqueue for publishing.
                self.edited_total += 1
                self._publish_to_social_media(video_output_dir, video_name)
            else:
                print("No output files found to organize")
                
        except Exception as e:
            print(f"Error organizing output files: {e}")
    
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
    parser = argparse.ArgumentParser(description='Monitor folder for new video files to process and publish.')
    parser.add_argument('--link', help='Optional URL of the original video/source to include in description')
    parser.add_argument('--publish-delay', type=int, default=None,
                        help='Minutes between publishes (overrides PUBLISH_INTERVAL_MINUTES; 0=immediate)')
    parser.add_argument('--no-dashboard', action='store_true',
                        help='Disable the periodic status dashboard')
    args = parser.parse_args()

    if args.link:
        print(f"🔗 Source link enabled: {args.link}")

    # Resolve publishing cadence: CLI flag overrides env, which defaults to 0 (immediate).
    publish_interval = args.publish_delay if args.publish_delay is not None else int(
        os.getenv("PUBLISH_INTERVAL_MINUTES", "0"))
    max_retries = int(os.getenv("PUBLISH_MAX_RETRIES", "3"))
    dashboard_interval = 0 if args.no_dashboard else int(os.getenv("DASHBOARD_INTERVAL_SECONDS", "60"))

    if publish_interval > 0:
        print(f"📤 Drip publishing: one video every {publish_interval} minute(s)")
    else:
        print("📤 Publishing: immediate (each edited video published in order)")

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
