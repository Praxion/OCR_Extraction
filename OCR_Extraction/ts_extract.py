import cv2
from PIL import Image
import pytesseract
import argparse
import os
import re
import logging
from datetime import datetime, timedelta
from collections import Counter, defaultdict

# ── Logging setup ──────────────────────────────────────────────────────────────
log_filename = f"ocr_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# ──────────────────────────────────────────────────────────────────────────────

JUMP_THRESHOLD_MINUTES = 10


def parseTimestamp(text):
    m = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", text)
    if m:
        try:
            dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            return True, dt
        except ValueError:
            return False, None
    return False, None


def getTimeFirstFrame(dt, frame_no, fps):
    offset = timedelta(seconds=frame_no / fps)
    return dt - offset


def GetTimestamp(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_secs = round(total_frames / fps, 2)
    filename = "temp_ocr_frame.bmp"
    frame_count = 0

    all_stamps = []
    ocr_failures = 0
    anomaly_frames = []
    baseline_stamp = None
    days_seen = set()

    # NEW: track every raw OCR timestamp seen, grouped by date
    raw_timestamps_by_day = defaultdict(set)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        roi = frame[0:int(height * 0.12), 0:int(width * 0.45)]
        cv2.imwrite(filename, roi)

        text = pytesseract.image_to_string(Image.open(filename))
        success, dt = parseTimestamp(text)

        if not success:
            ocr_failures += 1
            frame_count += 1
            continue

        # Track unique days and raw timestamps
        days_seen.add(dt.date())
        raw_timestamps_by_day[dt.date()].add(dt)  # NEW

        origin = getTimeFirstFrame(dt, frame_count, fps).replace(microsecond=0)
        all_stamps.append(origin)

        if baseline_stamp is None:
            baseline_stamp = origin

        diff = abs((origin - baseline_stamp).total_seconds())
        different_day = origin.date() != baseline_stamp.date()

        if diff > JUMP_THRESHOLD_MINUTES * 60 or different_day:
            anomaly_frames.append({
                "frame": frame_count,
                "raw_ts": dt,
                "origin": origin,
                "diff_minutes": round(diff / 60, 2),
                "different_day": different_day
            })

        frame_count += 1

    cap.release()
    if os.path.exists(filename):
        os.remove(filename)

    if not all_stamps:
        return False, None, 0, {
            "total_frames": frame_count,
            "ocr_failures": ocr_failures,
            "anomalies": [],
            "days_seen": [],
            "raw_timestamps_by_day": {},   # NEW
            "confidence": 0,
            "duration_secs": duration_secs,
            "fps": fps
        }

    vote_counts = Counter(all_stamps)
    best_stamp, vote_count = vote_counts.most_common(1)[0]
    confidence = round((vote_count / frame_count) * 100, 1)
    mismatch = frame_count - vote_count

    summary = {
        "total_frames": frame_count,
        "ocr_successes": len(all_stamps),
        "ocr_failures": ocr_failures,
        "confidence": confidence,
        "mismatch_frames": mismatch,
        "anomalies": anomaly_frames,
        "days_seen": sorted(days_seen),
        "raw_timestamps_by_day": {          # NEW
            day: sorted(ts_set)
            for day, ts_set in sorted(raw_timestamps_by_day.items())
        },
        "duration_secs": duration_secs,
        "fps": fps
    }

    return True, best_stamp, mismatch, summary


def logSummary(videoname, success, stamp, mismatch, summary):
    logger.info(f"{'='*60}")
    logger.info(f"VIDEO: {videoname}")
    logger.info(f"  Duration       : {summary['duration_secs']}s at {summary['fps']} FPS")
    logger.info(f"  Total frames   : {summary['total_frames']}")
    logger.info(f"  OCR successes  : {summary['ocr_successes']}")
    logger.info(f"  OCR failures   : {summary['ocr_failures']} frames")
    logger.info(f"  Confidence     : {summary['confidence']}%")
    logger.info(f"  Mismatch frames: {mismatch}")

    if success:
        logger.info(f"  Origin stamp   : {stamp.strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        logger.error(f"  Result         : FAILED — no timestamps found")

    # ── Detected timestamps grouped by day ────────────────────────────────────
    logger.info(f"")
    logger.info(f"  DETECTED TIMESTAMPS BY DAY:")
    ts_by_day = summary.get("raw_timestamps_by_day", {})
    if ts_by_day:
        for day, timestamps in ts_by_day.items():
            logger.info(f"    📅 {day} ({len(timestamps)} unique second(s) detected)")
            logger.info(f"       First : {timestamps[0].strftime('%H:%M:%S')}")
            logger.info(f"       Last  : {timestamps[-1].strftime('%H:%M:%S')}")
            logger.info(f"       Range : {str(timestamps[-1] - timestamps[0])} duration")
            # Print all unique timestamps for this day
            ts_strings = [t.strftime("%H:%M:%S") for t in timestamps]
            # Group into rows of 8 for readability
            for i in range(0, len(ts_strings), 8):
                logger.info(f"       {', '.join(ts_strings[i:i+8])}")
    else:
        logger.info(f"    No timestamps detected.")

    # ── Anomaly warnings ──────────────────────────────────────────────────────
    logger.info(f"")
    if summary['anomalies']:
        logger.warning(f"  ⚠️  ANOMALIES DETECTED — possible spliced/multi-session video!")
        for a in summary['anomalies']:
            if a['different_day']:
                logger.warning(
                    f"    Frame {a['frame']:>6}: timestamp {a['raw_ts']} is on a "
                    f"DIFFERENT DAY from baseline (origin: {a['origin']})"
                )
            else:
                logger.warning(
                    f"    Frame {a['frame']:>6}: timestamp {a['raw_ts']} is "
                    f"{a['diff_minutes']} min from baseline (origin: {a['origin']})"
                )
    else:
        logger.info(f"  ✅  No anomalies — video appears to be a single continuous session.")

    if len(summary['days_seen']) > 1:
        logger.warning(f"  ⚠️  MULTIPLE DATES FOUND: {[str(d) for d in summary['days_seen']]}")

    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True, help="Local directory containing videos")
    args = vars(ap.parse_args())

    video_dir = args["dir"]
    output_file = "stamp_file.txt"

    logger.info(f"Session started | Log: {log_filename} | Directory: {video_dir}")

    SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')

    video_list = [
        f for f in os.listdir(video_dir)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    if not video_list:
        logger.error(f"No video files found in {video_dir}")
    else:
        logger.info(f"Found {len(video_list)} video(s): {video_list}")

        logger.info(f"Found {len(video_list)} video(s) to process.\n")

        with open(output_file, "a+") as out:
            for videoname in video_list:
                full_path = os.path.join(video_dir, videoname)

                if not os.path.exists(full_path):
                    logger.warning(f"Skipping: {videoname} — file not found in {video_dir}")
                    continue

                success, stamp, mismatch, summary = GetTimestamp(full_path)
                logSummary(videoname, success, stamp, mismatch, summary)

                if success:
                    stamp_str = stamp.strftime("%Y-%m-%d %H:%M:%S")
                    anomaly_flag = " [ANOMALY DETECTED]" if summary['anomalies'] else ""
                    multiday_flag = " [MULTIPLE DAYS]" if len(summary['days_seen']) > 1 else ""
                    result = (
                        f"{videoname}, {stamp_str}, "
                        f"confidence={summary['confidence']}%, "
                        f"mismatch={mismatch} frames"
                        f"{anomaly_flag}{multiday_flag}\n"
                    )
                    out.write(result)
                else:
                    out.write(f"{videoname}, FAILED\n")

        logger.info("All videos processed.")