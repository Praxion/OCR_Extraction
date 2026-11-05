import cv2
from PIL import Image
import pytesseract
import os
import re
import logging
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── Logging Setup ─────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')


def parseTimestamp(text):
    """Searches for a YYYY-MM-DD HH:MM:SS timestamp pattern in OCR text."""
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            return True, dt
        except ValueError:
            return False, None
    return False, None


def extract_first_and_last_frame(video_path, output_dir, save_images=False, progress_callback=None):
    """
    Optimized function to extract the first and last frame timestamps and images.
    Scans the beginning few frames for the start time, and the ending frames for the end time.
    Images are only saved to disk if save_images is True.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file."

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = round(total_frames / fps, 2)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    os.makedirs(output_dir, exist_ok=True)
    temp_crop_file = "temp_ocr_crop.bmp"

    start_dt = None
    first_frame_img = None
    
    # 1. FIND START TIMESTAMP (Scan first 30 frames)
    if progress_callback:
        progress_callback(f"Scanning start frames for {video_name}...")

    for frame_idx in range(min(30, total_frames)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        roi = frame[0:int(height * 0.12), 0:int(width * 0.45)]
        cv2.imwrite(temp_crop_file, roi)

        text = pytesseract.image_to_string(Image.open(temp_crop_file))
        success, dt = parseTimestamp(text)

        if success:
            offset = timedelta(seconds=frame_idx / fps)
            start_dt = dt - offset
            first_frame_img = frame.copy()
            break

    # 2. FIND END TIMESTAMP (Scan last 30 frames backwards)
    end_dt = None
    last_frame_img = None

    if progress_callback:
        progress_callback(f"Scanning end frames for {video_name}...")

    for i in range(min(30, total_frames)):
        frame_idx = total_frames - 1 - i
        if frame_idx < 0:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        if last_frame_img is None:
            last_frame_img = frame.copy()

        height, width = frame.shape[:2]
        roi = frame[0:int(height * 0.12), 0:int(width * 0.45)]
        cv2.imwrite(temp_crop_file, roi)

        text = pytesseract.image_to_string(Image.open(temp_crop_file))
        success, dt = parseTimestamp(text)

        if success:
            offset = timedelta(seconds=(total_frames - 1 - frame_idx) / fps)
            end_dt = dt + offset
            break

    # Fallback if end timestamp OCR failed
    if start_dt and not end_dt:
        end_dt = start_dt + timedelta(seconds=duration_secs)

    if first_frame_img is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        if ret:
            first_frame_img = frame

    cap.release()
    if os.path.exists(temp_crop_file):
        os.remove(temp_crop_file)

    # 3. CONDITIONAL IMAGE SAVING (Only save if toggle switch is active)
    saved_files = []
    if save_images:
        if first_frame_img is not None:
            p1 = os.path.join(output_dir, f"{video_name}_first_frame.jpg")
            cv2.imwrite(p1, first_frame_img)
            saved_files.append(p1)

        if last_frame_img is not None:
            p2 = os.path.join(output_dir, f"{video_name}_last_frame.jpg")
            cv2.imwrite(p2, last_frame_img)
            saved_files.append(p2)

    if not start_dt:
        return False, {
            "video": video_name,
            "error": "Could not detect valid timestamp via OCR."
        }

    summary = {
        "video": video_name,
        "duration_secs": duration_secs,
        "fps": fps,
        "start_time": start_dt.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": end_dt.strftime('%Y-%m-%d %H:%M:%S') if end_dt else "Unknown",
        "saved_images": saved_files
    }

    return True, summary


class OCRAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame & Timestamp Extractor")
        self.root.geometry("750x600")
        self.root.minsize(650, 500)

        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_lbl = ttk.Label(main_frame, text="Video OCR First & Last Frame Extractor", font=("Arial", 14, "bold"))
        title_lbl.pack(pady=(0, 15))

        # Input Selection Frame
        input_frame = ttk.LabelFrame(main_frame, text=" Source Selection ", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Video Path / Folder:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.input_entry = ttk.Entry(input_frame, width=50)
        self.input_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(input_frame, text="Browse Folder", command=self.browse_input_dir).grid(row=0, column=2, padx=5, pady=5)
        input_frame.columnconfigure(1, weight=1)

        # Output Selection Frame
        output_frame = ttk.LabelFrame(main_frame, text=" Output Destination ", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="Save Results To:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.output_entry = ttk.Entry(output_frame, width=50)
        self.output_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(output_frame, text="Browse", command=self.browse_output).grid(row=0, column=2, padx=5, pady=5)
        output_frame.columnconfigure(1, weight=1)

        # Testing Options / Toggle Switch Frame
        options_frame = ttk.LabelFrame(main_frame, text=" Testing Options ", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        self.save_images_var = tk.BooleanVar(value=False)
        self.save_images_check = ttk.Checkbutton(
            options_frame, 
            text="Save first and last frame images to disk (Testing Mode)", 
            variable=self.save_images_var
        )
        self.save_images_check.pack(anchor="w", padx=5, pady=2)

        # Action Button Frame
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.X, pady=5)

        self.run_btn = ttk.Button(action_frame, text="Start Extraction", command=self.start_extraction)
        self.run_btn.pack(side=tk.RIGHT, padx=5)

        # Progress Bar
        self.progress = ttk.Progressbar(main_frame, orient="horizontal", mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=10)

        # Log / Console Output Text Box
        log_frame = ttk.LabelFrame(main_frame, text=" Activity Log ", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = tk.Text(log_frame, height=10, wrap=tk.WORD, bg="#f8f9fa", font=("Consolas", 9))
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.log("Ready. Select a video file or folder to begin.")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        logger.info(message)

    def browse_input_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, d)

    def browse_input_file(self):
        f = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv")])
        if f:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, f)

    def browse_output(self):
        d = filedialog.askdirectory()
        if d:
            self.output_entry.delete(0, tk.END)
            self.output_entry.insert(0, d)

    def start_extraction(self):
        input_path = self.input_entry.get().strip()
        output_path = self.output_entry.get().strip()
        save_images_flag = self.save_images_var.get()

        if not input_path or not output_path:
            messagebox.showerror("Error", "Please specify both an input video/folder and an output destination.")
            return

        video_files = []
        if os.path.isfile(input_path):
            video_files = [input_path]
        elif os.path.isdir(input_path):
            video_files = [
                os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.lower().endswith(SUPPORTED_EXTENSIONS)
            ]

        if not video_files:
            messagebox.showwarning("Warning", "No valid video files found in the specified path.")
            return

        self.run_btn.config(state=tk.DISABLED)
        self.progress.start(10)
        self.root.update_idletasks()

        # ── DYNAMIC OUTPUT FILE NAMING MATCHING RUN DATE ──────────────────────
        run_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        output_txt_file = os.path.join(output_path, f"extracted_timestamps_{run_date_str}.txt")
        # ──────────────────────────────────────────────────────────────────────

        success_count = 0

        try:
            with open(output_txt_file, "a", encoding="utf-8") as out_f:
                for v_path in video_files:
                    v_name = os.path.basename(v_path)
                    self.log(f"Processing: {v_name}...")
                    
                    success, result = extract_first_and_last_frame(
                        v_path, output_path, 
                        save_images=save_images_flag, 
                        progress_callback=lambda msg: self.log(f"   ↳ {msg}")
                    )

                    if success:
                        success_count += 1
                        self.log(f"   ✅ Success! Start: {result['start_time']} | End: {result['end_time']}")
                        if save_images_flag:
                            self.log(f"      📸 Snapshots saved to destination folder.")
                        out_f.write(f"{result['video']}, Start: {result['start_time']}, End: {result['end_time']}, Duration: {result['duration_secs']}s\n")
                    else:
                        self.log(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                        out_f.write(f"{result.get('video', v_name)}, FAILED\n")

            self.log(f"\n✨ Finished! Successfully processed {success_count}/{len(video_files)} video(s).")
            self.log(f"📁 Results saved to: {output_txt_file}")
            messagebox.showinfo("Complete", f"Processing finished!\nResults saved to:\n{output_txt_file}")

        except Exception as e:
            self.log(f"❌ Critical Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")

        finally:
            self.progress.stop()
            self.run_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRAppGUI(root)
    root.mainloop()