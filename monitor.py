import time
import os
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv

class NewVideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".mp4", ".mov", ".avi", ".mkv")):
            print(f"New video detected: {event.src_path}")
            self.process_video(event.src_path)

    def process_video(self, video_path):
        print(f"Processing video: {video_path}")
        try:
            # We need to run reelsfy.py from within its directory
            reelsfy_path = os.path.join(os.path.dirname(__file__), "reels-clips-automator", "reelsfy.py")
            reelsfy_dir = os.path.dirname(reelsfy_path)
            subprocess.run(["python", reelsfy_path, "--file", video_path], check=True, cwd=reelsfy_dir)
            print(f"Finished processing video: {video_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing video: {e}")

def main():
    # Load environment variables from the root .env file
    load_dotenv()

    video_folder = os.getenv("VIDEO_FOLDER")
    if not video_folder:
        print("Error: VIDEO_FOLDER not set in .env file.")
        exit(1)

    # Expand user path (e.g., ~)
    video_folder = os.path.expanduser(video_folder)

    print(f"Monitoring folder: {video_folder}")

    event_handler = NewVideoHandler()
    observer = Observer()
    observer.schedule(event_handler, video_folder, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    main()
