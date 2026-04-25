import cv2
import numpy as np
import time
from collections import deque
from ultralytics import YOLO

MODEL_PATH = "yolov8n.pt"

VIDEO_PATHS = {
    "A": "traffic1.mp4",
    "B": "traffic2.mp4",
    "C": "traffic3.mp4",
    "D": "traffic4.mp4"
}

CONFIDENCE = 0.20

FRAME_SKIP = 3
YELLOW_UPDATE_INTERVAL = 6

LANE_FRAME_WIDTH = 640
LANE_FRAME_HEIGHT = 360
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

SMOOTHING_WINDOW = 12

VEHICLE_CLASSES = [2, 3, 5, 7]

VEHICLE_WEIGHTS = {
    2: 2,
    3: 1,
    5: 4,
    7: 4
}

LOW_THRESHOLD = 10
MEDIUM_THRESHOLD = 22

MIN_GREEN_TIME = 8
MID_GREEN_TIME = 14
MAX_GREEN_TIME = 22

YELLOW_TIME = 3
ALL_RED_TIME = 1

WAIT_FACTOR = 0.8
COOLDOWN_PENALTY_FACTOR = 1.0
GREEN_COOLDOWN_SECONDS = 10
SAME_LANE_TOLERANCE = 2.0
OCCUPANCY_WEIGHT = 30.0

ROI_POLYGONS = {
    "A": np.array([(80, 100), (610, 100), (625, 335), (60, 335)], dtype=np.int32),
    "B": np.array([(80, 100), (610, 100), (625, 335), (60, 335)], dtype=np.int32),
    "C": np.array([(80, 100), (610, 100), (625, 335), (60, 335)], dtype=np.int32),
    "D": np.array([(80, 100), (610, 100), (625, 335), (60, 335)], dtype=np.int32)
}

def point_inside_roi(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (int(x), int(y)), False) >= 0

def get_green_time(final_load):
    if final_load < LOW_THRESHOLD:
        return MIN_GREEN_TIME
    elif final_load < MEDIUM_THRESHOLD:
        return MID_GREEN_TIME
    return MAX_GREEN_TIME

def create_blank_lane_frame(lane_name):
    frame = np.zeros((LANE_FRAME_HEIGHT, LANE_FRAME_WIDTH, 3), dtype=np.uint8)
    cv2.putText(frame, f"Lane {lane_name} Video Ended", (150, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

def draw_signal_set(frame, x, y, active_state, title):
    radius = 14

    red_color = (0, 0, 255) if active_state == "RED" else (60, 60, 60)
    yellow_color = (0, 255, 255) if active_state == "YELLOW" else (60, 60, 60)
    green_color = (0, 255, 0) if active_state == "GREEN" else (60, 60, 60)

    cv2.rectangle(frame, (x - 18, y - 28), (x + 110, y + 108), (30, 30, 30), -1)
    cv2.rectangle(frame, (x - 18, y - 28), (x + 110, y + 108), (200, 200, 200), 2)

    cv2.putText(frame, title, (x - 5, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.circle(frame, (x, y + 12), radius, red_color, -1)
    cv2.circle(frame, (x, y + 46), radius, yellow_color, -1)
    cv2.circle(frame, (x, y + 80), radius, green_color, -1)

def draw_lane_status(frame, lane_name, signal_state):
    color_map = {
        "RED": (0, 0, 255),
        "YELLOW": (0, 255, 255),
        "GREEN": (0, 255, 0)
    }
    color = color_map.get(signal_state, (255, 255, 255))
    cv2.putText(frame, f"Lane {lane_name}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.putText(frame, f"Signal: {signal_state}", (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2)

def compute_roi_area(roi_polygon):
    return max(1.0, cv2.contourArea(roi_polygon.astype(np.float32)))

def compute_box_area_inside_roi(box, roi_polygon, frame_shape):
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    if x2 <= x1 or y2 <= y1:
        return 0

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [roi_polygon], 255)
    roi_crop = mask[y1:y2, x1:x2]
    return cv2.countNonZero(roi_crop)

def choose_next_lane(final_loads, waiting_times, green_cooldowns, current_green_lane):
    priority_scores = {}

    for lane in final_loads:
        penalty = green_cooldowns[lane] * COOLDOWN_PENALTY_FACTOR
        priority_scores[lane] = final_loads[lane] + waiting_times[lane] * WAIT_FACTOR - penalty

    if current_green_lane is not None:
        other_lanes = [lane for lane in priority_scores if lane != current_green_lane]
        if other_lanes:
            best_other = max(other_lanes, key=lambda lane: priority_scores[lane])
            if priority_scores[best_other] >= priority_scores[current_green_lane] - SAME_LANE_TOLERANCE:
                return best_other, priority_scores

    best_lane = max(priority_scores, key=priority_scores.get)
    return best_lane, priority_scores

model = YOLO(MODEL_PATH)
class_names = model.names

caps = {}
for lane, path in VIDEO_PATHS.items():
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: Could not open {path}")
        raise SystemExit
    caps[lane] = cap

frame_index = 0
ended_lanes = set()

lane_frames = {}
last_raw_frames = {}

lane_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
lane_weighted_loads = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
lane_occupancies = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
avg_weighted_loads = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
avg_occupancies = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
final_loads = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

waiting_times = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
green_cooldowns = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
priority_scores = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}

weighted_history = {
    "A": deque(maxlen=SMOOTHING_WINDOW),
    "B": deque(maxlen=SMOOTHING_WINDOW),
    "C": deque(maxlen=SMOOTHING_WINDOW),
    "D": deque(maxlen=SMOOTHING_WINDOW)
}

occupancy_history = {
    "A": deque(maxlen=SMOOTHING_WINDOW),
    "B": deque(maxlen=SMOOTHING_WINDOW),
    "C": deque(maxlen=SMOOTHING_WINDOW),
    "D": deque(maxlen=SMOOTHING_WINDOW)
}

roi_areas = {lane: compute_roi_area(ROI_POLYGONS[lane]) for lane in ROI_POLYGONS}

controller_state = "GREEN"
current_green_lane = "A"
next_green_lane = "A"
state_start_time = time.time()
current_green_duration = MIN_GREEN_TIME
last_update_time = time.time()

while True:
    frame_index += 1
    now = time.time()
    dt = now - last_update_time
    last_update_time = now

    for lane in green_cooldowns:
        green_cooldowns[lane] = max(0.0, green_cooldowns[lane] - dt)

    signal_states = {"A": "RED", "B": "RED", "C": "RED", "D": "RED"}

    if controller_state == "GREEN":
        signal_states[current_green_lane] = "GREEN"
    elif controller_state == "YELLOW":
        signal_states[current_green_lane] = "YELLOW"

    for lane in ["A", "B", "C", "D"]:
        if lane in ended_lanes:
            lane_frames[lane] = create_blank_lane_frame(lane)
            continue

        should_advance = False
        if signal_states[lane] == "GREEN":
            should_advance = True
        elif signal_states[lane] == "YELLOW" and frame_index % YELLOW_UPDATE_INTERVAL == 0:
            should_advance = True

        if lane not in last_raw_frames:
            ret, frame = caps[lane].read()
            if not ret:
                ended_lanes.add(lane)
                lane_frames[lane] = create_blank_lane_frame(lane)
                continue
            frame = cv2.resize(frame, (LANE_FRAME_WIDTH, LANE_FRAME_HEIGHT))
            last_raw_frames[lane] = frame.copy()

        elif should_advance:
            ret, frame = caps[lane].read()
            if not ret:
                ended_lanes.add(lane)
                lane_frames[lane] = create_blank_lane_frame(lane)
                continue
            frame = cv2.resize(frame, (LANE_FRAME_WIDTH, LANE_FRAME_HEIGHT))
            last_raw_frames[lane] = frame.copy()
        else:
            frame = last_raw_frames[lane].copy()

        display_frame = frame.copy()
        roi = ROI_POLYGONS[lane]
        cv2.polylines(display_frame, [roi], True, (255, 0, 0), 2)
        draw_lane_status(display_frame, lane, signal_states[lane])

        run_detection = should_advance and (frame_index % FRAME_SKIP == 0)

        if run_detection:
            current_visible_vehicles = 0
            current_weighted_load = 0.0
            current_box_area = 0.0

            results = model(frame, conf=CONFIDENCE, verbose=False)

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls = int(box.cls[0].item())
                    if cls not in VEHICLE_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    if not point_inside_roi(cx, cy, roi):
                        continue

                    current_visible_vehicles += 1
                    current_weighted_load += VEHICLE_WEIGHTS.get(cls, 1)
                    current_box_area += compute_box_area_inside_roi((x1, y1, x2, y2), roi, frame.shape)

                    label = class_names[cls]
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(display_frame, (cx, cy), 3, (0, 255, 255), -1)
                    cv2.putText(display_frame, label, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

            occupancy_ratio = min(1.0, current_box_area / roi_areas[lane])

            lane_counts[lane] = current_visible_vehicles
            lane_weighted_loads[lane] = current_weighted_load
            lane_occupancies[lane] = occupancy_ratio

            weighted_history[lane].append(current_weighted_load)
            occupancy_history[lane].append(occupancy_ratio)

            avg_weighted_loads[lane] = (
                sum(weighted_history[lane]) / len(weighted_history[lane])
                if len(weighted_history[lane]) > 0 else 0.0
            )
            avg_occupancies[lane] = (
                sum(occupancy_history[lane]) / len(occupancy_history[lane])
                if len(occupancy_history[lane]) > 0 else 0.0
            )

            final_loads[lane] = avg_weighted_loads[lane] + avg_occupancies[lane] * OCCUPANCY_WEIGHT

        cv2.putText(display_frame, f"Vehicles: {lane_counts[lane]}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Load: {lane_weighted_loads[lane]:.1f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Occupancy: {lane_occupancies[lane]*100:.1f}%", (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Final Load: {final_loads[lane]:.1f}", (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Waiting: {waiting_times[lane]:.1f}s", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2)

        lane_frames[lane] = display_frame

    if len(ended_lanes) == 4:
        print("All videos ended.")
        break

    for lane in waiting_times:
        if lane == current_green_lane and controller_state == "GREEN":
            waiting_times[lane] = 0.0
        else:
            waiting_times[lane] += dt

    elapsed = now - state_start_time

    if controller_state == "GREEN":
        current_green_duration = get_green_time(final_loads[current_green_lane])

        if elapsed >= current_green_duration:
            next_green_lane, priority_scores = choose_next_lane(
                final_loads,
                waiting_times,
                green_cooldowns,
                current_green_lane
            )

            if next_green_lane != current_green_lane:
                controller_state = "YELLOW"
                state_start_time = time.time()
            else:
                state_start_time = time.time()

    elif controller_state == "YELLOW":
        if elapsed >= YELLOW_TIME:
            controller_state = "ALL_RED"
            state_start_time = time.time()

    elif controller_state == "ALL_RED":
        if elapsed >= ALL_RED_TIME:
            current_green_lane = next_green_lane
            waiting_times[current_green_lane] = 0.0
            green_cooldowns[current_green_lane] = GREEN_COOLDOWN_SECONDS
            controller_state = "GREEN"
            state_start_time = time.time()

    signal_states = {"A": "RED", "B": "RED", "C": "RED", "D": "RED"}

    if controller_state == "GREEN":
        signal_states[current_green_lane] = "GREEN"
    elif controller_state == "YELLOW":
        signal_states[current_green_lane] = "YELLOW"

    if controller_state == "GREEN":
        time_left = max(0, int(current_green_duration - (time.time() - state_start_time)))
    elif controller_state == "YELLOW":
        time_left = max(0, int(YELLOW_TIME - (time.time() - state_start_time)))
    else:
        time_left = max(0, int(ALL_RED_TIME - (time.time() - state_start_time)))

    top_row = np.hstack([lane_frames["A"], lane_frames["B"]])
    bottom_row = np.hstack([lane_frames["C"], lane_frames["D"]])
    dashboard = np.vstack([top_row, bottom_row])
    dashboard = cv2.resize(dashboard, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

    overlay = dashboard.copy()
    cv2.rectangle(overlay, (860, 0), (1280, 720), (20, 20, 20), -1)
    dashboard = cv2.addWeighted(overlay, 0.65, dashboard, 0.35, 0)

    draw_signal_set(dashboard, 910, 40, signal_states["A"], "Lane A")
    draw_signal_set(dashboard, 1060, 40, signal_states["B"], "Lane B")
    draw_signal_set(dashboard, 910, 190, signal_states["C"], "Lane C")
    draw_signal_set(dashboard, 1060, 190, signal_states["D"], "Lane D")

    cv2.putText(dashboard, f"State: {controller_state}", (900, 340),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(dashboard, f"Green Lane: {current_green_lane}", (900, 375),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)
    cv2.putText(dashboard, f"Timer: {time_left}s", (900, 410),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (0, 255, 255), 2)

    y = 470
    for lane in ["A", "B", "C", "D"]:
        score = final_loads[lane] + waiting_times[lane] * WAIT_FACTOR - green_cooldowns[lane] * COOLDOWN_PENALTY_FACTOR
        cv2.putText(
            dashboard,
            f"{lane} | Load:{final_loads[lane]:.1f} Occ:{avg_occupancies[lane]*100:.0f}% Wait:{waiting_times[lane]:.0f}s",
            (885, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2
        )
        y += 36

    cv2.putText(dashboard, "ESC = Exit", (1080, 690),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)

    cv2.namedWindow("Fast 4-Lane Smart Traffic Controller", cv2.WINDOW_NORMAL)
    cv2.imshow("Fast 4-Lane Smart Traffic Controller", dashboard)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

for cap in caps.values():
    cap.release()

cv2.destroyAllWindows()