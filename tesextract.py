import cv2
from PIL import Image
import pytesseract
import os
import re
from collections import Counter
import logging
import sys
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

# ── Logging Setup (UTF-8 safe for Windows consoles) ───────────────────────────
os.makedirs("logs", exist_ok=True)
log_filename = os.path.join("logs", f"ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

file_handler = logging.FileHandler(log_filename, encoding='utf-8')
stream_handler = logging.StreamHandler(sys.stdout)
if hasattr(stream_handler.stream, "reconfigure"):
    try:
        stream_handler.stream.reconfigure(encoding='utf-8')
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[file_handler, stream_handler]
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')


TS_RE = re.compile(r"(\d{4})\D(\d{2})\D(\d{2})\D+(\d{2})\D(\d{2})\D(\d{2})")


def parseTimestamp(text):
    """Finds a YYYY-MM-DD HH:MM:SS timestamp in OCR text, tolerating odd separators."""
    m = TS_RE.search(text)
    if not m:
        return False, None
    try:
        dt = datetime(*map(int, m.groups()))
    except ValueError:
        return False, None
    if not (2000 <= dt.year <= datetime.now().year + 1):
        return False, None
    return True, dt
FRAMES_TO_READ = 7      # max consecutive frames to sample per video
AGREE_NEEDED = 2        # identical reads required before a timestamp is accepted
# (height_fraction, width_fraction) crops from the top-left corner, tightest first
ROI_CANDIDATES = [(0.07, 0.35), (0.10, 0.45), (0.15, 0.75)]
_winner = None          # (crop_index, label, psm) that last produced an agreed read


def make_variants(roi):
    """Returns {label: image}: preprocessing variants of the cropped region."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    out = {}
    for scale in (1, 3):
        g = gray if scale == 1 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        out[f"gray x{scale}"] = g
        out[f"gray-inv x{scale}"] = cv2.bitwise_not(g)
        for t in (150, 180, 200, 220, 240):
            _, m = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
            out[f"white>{t} x{scale}"] = cv2.bitwise_not(m)   # black text on white
    return out


def _crop(frame, crop_idx):
    h, w = frame.shape[:2]
    hf, wf = ROI_CANDIDATES[crop_idx]
    return frame[0:int(h * hf), 0:int(w * wf)]


def _ocr(img, psm):
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_REPLICATE)
    return pytesseract.image_to_string(img, config=f"--psm {psm}")


def ocr_timestamp(frame):
    """Returns (datetime or None, config_used or None, raw_text). Tries the cached config first."""
    if _winner:
        ci, label, psm = _winner
        text = _ocr(make_variants(_crop(frame, ci))[label], psm)
        ok, dt = parseTimestamp(text)
        if ok:
            return dt, _winner, text

    text = ""
    for ci in range(len(ROI_CANDIDATES)):
        for label, img in make_variants(_crop(frame, ci)).items():
            for psm in (7, 6):
                text = _ocr(img, psm)
                ok, dt = parseTimestamp(text)
                if ok:
                    return dt, (ci, label, psm), text
    return None, None, text


def extract_first_frame_only(video_path, output_dir, save_images=False, progress_callback=None):
    """Testing Mode: reads up to FRAMES_TO_READ frames and accepts a timestamp only when frames agree."""
    global _winner
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    if progress_callback:
        progress_callback(f"Opening video stream for {video_name}...")

    cap = cv2.VideoCapture(video_path)
    first_frame = None
    reads = []          # one parsed datetime per frame that OCR could read
    accepted = None     # (datetime, agree_count, frames_used, config)
    try:
        if not cap.isOpened():
            return False, {"video": video_name, "error": "Could not open video file (moov atom or header issue)."}
        for i in range(FRAMES_TO_READ):
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if first_frame is None:
                first_frame = frame
            dt, cfg, _text = ocr_timestamp(frame)
            if dt is None:
                continue
            reads.append((dt, cfg))
            value, count = Counter(d for d, _ in reads).most_common(1)[0]
            if count >= AGREE_NEEDED:
                accepted = (value, count, i + 1, cfg)
                break
    finally:
        cap.release()

    if first_frame is None:
        return False, {"video": video_name, "error": "Failed to read the first frame."}

    if accepted is None:
        seen = [d.strftime('%H:%M:%S') for d, _ in reads] or "nothing parseable"
        return False, {"video": video_name,
                       "error": f"No {AGREE_NEEDED} frames agreed. Reads: {seen}"}

    dt, count, frames_used, cfg = accepted
    _winner = cfg       # try this config first on the next video
    ci, label, psm = cfg

    saved_files = []
    if save_images:
        p1 = os.path.join(output_dir, f"{video_name}_first_frame.jpg")
        cv2.imwrite(p1, first_frame)
        saved_files.append(p1)

    summary = {
        "video": video_name,
        "start_time": dt.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": "Testing Mode (First Frame Only)",
        "ocr_method": f"{count}/{frames_used} frames agreed; crop {ci}, {label}, psm {psm}",
        "saved_images": saved_files
    }
    return True, summary

class OCRAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Frame & Timestamp Extractor (OpenCV Engine)")
        self.root.geometry("750x600")
        self.root.minsize(650, 500)

        style = ttk.Style()
        style.theme_use("clam")

        main_frame = ttk.Frame(root, padding="15")
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_lbl = ttk.Label(main_frame, text="Video OCR Extractor (Streamlined OpenCV)", font=("Arial", 14, "bold"))
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
        options_frame = ttk.LabelFrame(main_frame, text=" Testing Options & Performance ", padding="10")
        options_frame.pack(fill=tk.X, pady=5)

        self.save_images_var = tk.BooleanVar(value=True)
        self.save_images_check = ttk.Checkbutton(
            options_frame, 
            text="Save first frame snapshot to disk (Testing Mode)", 
            variable=self.save_images_var
        )
        self.save_images_check.pack(anchor="w", padx=5, pady=2)

        info_lbl = ttk.Label(options_frame, text="ℹ️ Configured to extract only the first frame via OpenCV for rapid baseline testing.", font=("Arial", 8, "italic"), foreground="gray")
        info_lbl.pack(anchor="w", padx=5, pady=2)

        # Action Button Frame
        action_frame = ttk.Frame(main_frame, padding="10")
        action_frame.pack(fill=tk.X, pady=5)

        self.run_btn = ttk.Button(action_frame, text="Start Standard Testing", command=self.start_extraction)
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

        self.log("Ready. Using streamlined OpenCV decoding workflow.")

    def log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        try:
            logger.info(message)
        except UnicodeEncodeError:
            logger.info(message.encode('ascii', 'ignore').decode('ascii'))

    def browse_input_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(0, d)

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

        run_date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        os.makedirs(output_path, exist_ok=True)
        output_txt_file = os.path.join(output_path, f"extracted_timestamps_test_{run_date_str}.txt")

        success_count = 0
        start_time_total = datetime.now()

        try:
            with open(output_txt_file, "a", encoding="utf-8") as out_f:
                for v_path in video_files:
                    v_name = os.path.basename(v_path)
                    self.log(f"Processing (First Frame Test): {v_name}...")
                    
                    try:
                        success, result = extract_first_frame_only(
                            v_path, output_path,
                            save_images=save_images_flag,
                            progress_callback=lambda msg: self.log(f"   [->] {msg}")
                        )
                    except Exception as e:
                        success, result = False, {"video": v_name, "error": f"{type(e).__name__}: {e}"}

                    if success:
                        success_count += 1
                        self.log(f"   [SUCCESS] Detected Timestamp: {result['start_time']} (via {result['ocr_method']})")
                        if save_images_flag:
                            self.log(f"      [SNAPSHOT] First frame snapshot saved.")
                        out_f.write(f"{result['video']}, First Frame Timestamp: {result['start_time']}\n")
                    else:
                        err_msg = result.get('error', 'Unknown error') if isinstance(result, dict) else str(result)
                        self.log(f"   [FAILED] {err_msg}")
                        video_key = result.get('video', v_name) if isinstance(result, dict) else v_name
                        out_f.write(f"{video_key}, FAILED: {err_msg}\n")

            elapsed_ms = (datetime.now() - start_time_total).total_seconds()
            self.log(f"\nFinished! Processed {success_count}/{len(video_files)} video(s) in {elapsed_ms:.2f}s total.")
            self.log(f"Results saved to: {output_txt_file}")
            messagebox.showinfo("Complete", f"Standard testing finished!\nResults saved to:\n{output_txt_file}")

        except Exception as e:
            self.log(f"Critical Error: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during processing:\n{str(e)}")

        finally:
            self.progress.stop()
            self.run_btn.config(state=tk.NORMAL)


if __name__ == "__main__":
    root = tk.Tk()
    app = OCRAppGUI(root)
    root.mainloop()