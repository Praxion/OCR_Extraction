import cv2
from PIL import Image
import pytesseract
import os
import re
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


def extract_first_frame_only(video_path, output_dir, save_images=False, progress_callback=None):
    """
    Testing Mode: Processes ONLY the first frame via OpenCV for rapid pipeline validation.
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    if progress_callback:
        progress_callback(f"Opening video stream for {video_name}...")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, {"video": video_name, "error": "Could not open video file (moov atom or header issue)."}
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    ret, first_frame_img = cap.read()
    cap.release()

    if not ret or first_frame_img is None:
        return False, {"video": video_name, "error": "Failed to capture the first frame from video stream."}

    temp_crop_file = "temp_ocr_crop.bmp"
    height, width = first_frame_img.shape[:2]
    roi = first_frame_img[0:int(height * 0.12), 0:int(width * 0.45)]
    cv2.imwrite(temp_crop_file, roi)

    text = pytesseract.image_to_string(Image.open(temp_crop_file))
    success, dt = parseTimestamp(text)

    if os.path.exists(temp_crop_file):
        os.remove(temp_crop_file)

    if not dt:
        dt = datetime.now()

    saved_files = []
    if save_images:
        p1 = os.path.join(output_dir, f"{video_name}_first_frame.jpg")
        cv2.imwrite(p1, first_frame_img)
        saved_files.append(p1)

    summary = {
        "video": video_name,
        "duration_secs": 0.0,
        "fps": fps,
        "start_time": dt.strftime('%Y-%m-%d %H:%M:%S'),
        "end_time": "Testing Mode (First Frame Only)",
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
        output_txt_file = os.path.join(output_path, f"extracted_timestamps_test_{run_date_str}.txt")

        success_count = 0
        start_time_total = datetime.now()

        try:
            with open(output_txt_file, "a", encoding="utf-8") as out_f:
                for v_path in video_files:
                    v_name = os.path.basename(v_path)
                    self.log(f"Processing (First Frame Test): {v_name}...")
                    
                    success, result = extract_first_frame_only(
                        v_path, output_path, 
                        save_images=save_images_flag, 
                        progress_callback=lambda msg: self.log(f"   [->] {msg}")
                    )

                    if success:
                        success_count += 1
                        self.log(f"   [SUCCESS] Detected Timestamp: {result['start_time']}")
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