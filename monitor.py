import time
import os
import subprocess
import shutil
import threading
from pathlib import Path
from datetime import datetime
from watchdog.observers import Observer
from watchdog.observers.polling import PollingObserver
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

class NewVideoHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing_files = set()  # Track files currently being processed
        self.lock = threading.Lock()
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".wmv", ".m4v")):
            file_path = Path(event.src_path)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] New video detected: {file_path.name}")
            
            # Start monitoring this file in a separate thread
            threading.Thread(
                target=self._monitor_and_process_file,
                args=(file_path,),
                daemon=True
            ).start()
    
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
                result = subprocess.run(
                    ["python", str(reelsfy_path), "-f", str(video_path), "--output-dir", "../"+str(video_output_dir)], 
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
            
            files_to_move = [
                ("*.mp4", "final_video.mp4"),
                ("*.srt", "subtitles.srt"), 
                ("content.txt", "content.txt"),
                ("transcript.txt", "transcript.txt"),
                ("description.txt", "description.txt"),
            ]
            
            moved_files = []
            
            if reelsfy_outputs_dir.exists():
                import glob
                for pattern, dest_name in files_to_move:
                    if pattern.startswith("*"):
                        # Handle glob patterns
                        matches = list(reelsfy_outputs_dir.glob(pattern))
                        for match in matches:
                            if match.is_file():
                                dest_path = video_output_dir / dest_name.replace(".mp4", f"_{match.stem}.mp4").replace(".srt", f"_{match.stem}.srt")
                                shutil.move(str(match), str(dest_path))
                                moved_files.append(dest_path.name)
                                print(f"Moved {match.name} to {dest_path.name}")
                                break  # Only move the first match
                    else:
                        # Handle specific filenames
                        source_path = reelsfy_outputs_dir / pattern
                        if source_path.exists():
                            dest_path = video_output_dir / dest_name
                            shutil.move(str(source_path), str(dest_path))
                            moved_files.append(dest_name)
                            print(f"Moved {pattern} to {dest_name}")
            
            if moved_files:
                print(f"Organized {len(moved_files)} files in {video_output_dir}")
                
                # Integrate social media publishing
                self._publish_to_social_media(video_output_dir, video_name)
            else:
                print("No output files found to organize")
                
        except Exception as e:
            print(f"Error organizing output files: {e}")
    
    def _publish_to_social_media(self, video_output_dir, video_name):
        """Publish video to social media platforms via Postiz API"""
        try:
            print("📱 Starting social media publishing...")
            
            # Import and initialize publisher
            from social_media_publisher import PostizPublisher
            publisher = PostizPublisher(str(video_output_dir))
            
            # Start async publishing (non-blocking)
            publisher.publish_async()
            print("📱 Social media publishing started in background")
            
        except ImportError:
            print("📱 Social media publisher not available (module not found)")
        except ValueError as e:
            print(f"📱 Social media publishing configuration error: {e}")
        except Exception as e:
            print(f"📱 Social media publishing error: {e}")

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
    print(f"👀 Monitoring: {video_folder_path.absolute()}")
    print(f"📁 Output to: {output_folder.absolute()}")
    print(f"🔍 Observer type: {observer_type}")
    print("🎯 Supported formats: .mp4, .mov, .avi, .mkv, .webm, .flv, .wmv, .m4v")

    # Create directories if they don't exist
    video_folder_path.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(exist_ok=True)

    event_handler = NewVideoHandler()
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
    observer.join()
    print("👋 Monitor stopped")

if __name__ == "__main__":
    main()
