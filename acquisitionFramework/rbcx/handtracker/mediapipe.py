# mediapipe_handtracker.py

import time
import threading
import platform

from copy import deepcopy

import cv2
import numpy as np
import mediapipe as mp


class MediaPipeHandTracker:
    """
    Threaded MediaPipe Hands, OpenCV overlay and 3D joint angles estimation and display
    Call start() to launch the capture/processing thread
    Optionally shows one OpenCV window (set show_window=True/False)
    Get angles anytime with get_mediapipe_angles()
    Call stop() to stop the thread

    Parameters
    ----------
    camera_index : int: OpenCV camera index
    mirror : bool: if True, mirror the display (processing remains non-mirrored)
    show_window : bool: if True, show a live window from the internal thread. 
                        you can also access the image simply with 'get_last_frame()' without showing the window
    MediaPipe Hands options: max_num_hands, model_complexity (1 for best model, 0 for faster), min_detection_confidence, min_tracking_confidence
    """

    def __init__(
        self,
        camera_index: int = 0,
        window_name: str = "Hand Tracking",
        mirror: bool = False,
        show_window: bool = True,
        max_num_hands: int = 2,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        use_2D_coord_for_angles=True,
        capture_backend=None,
    ):
        self.camera_index = camera_index
        self.window_name = window_name
        self.mirror = mirror
        self.show_window = show_window

        self.max_num_hands = max_num_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.capture_backend = (
            capture_backend
            if capture_backend is not None
            else (cv2.CAP_AVFOUNDATION if platform.system() == "Darwin" else cv2.CAP_ANY)
        )

        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._latest_right_angles = {
            'timestamp': time.time(),
            'angles': {
                'Thumb_CMC': 0, 'Thumb_MCP': 0, 'Thumb_IP': 0,
                'Index_MCP': 0, 'Index_PIP': 0, 'Index_DIP': 0,
                'Middle_MCP': 0, 'Middle_PIP': 0, 'Middle_DIP': 0,
                'Ring_MCP': 0, 'Ring_PIP': 0, 'Ring_DIP': 0,
                'Pinky_MCP': 0, 'Pinky_PIP': 0, 'Pinky_DIP': 0
            },
            'landmarks': None,          # np.ndarray of shape (21, 3)
            'landmarks_type': None      # 'world' or 'normalized'
        }
        self._latest_left_angles = {
            'timestamp': time.time(),
            'angles': {
                'Thumb_CMC': 0, 'Thumb_MCP': 0, 'Thumb_IP': 0,
                'Index_MCP': 0, 'Index_PIP': 0, 'Index_DIP': 0,
                'Middle_MCP': 0, 'Middle_PIP': 0, 'Middle_DIP': 0,
                'Ring_MCP': 0, 'Ring_PIP': 0, 'Ring_DIP': 0,
                'Pinky_MCP': 0, 'Pinky_PIP': 0, 'Pinky_DIP': 0
            },
            'landmarks': None,          # np.ndarray of shape (21, 3)
            'landmarks_type': None      # 'world' or 'normalized'
        }

        self.joint_names = [
            'Thumb_CMC', 'Thumb_MCP', 'Thumb_IP',
            'Index_MCP', 'Index_PIP', 'Index_DIP',
            'Middle_MCP', 'Middle_PIP', 'Middle_DIP',
            'Ring_MCP', 'Ring_PIP', 'Ring_DIP',
            'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP'
        ]

        self._latest_frame = None
        self._running = False
        self.use_2D_coord_for_angles = use_2D_coord_for_angles

        # New: OpenCV GUI must be managed on the main thread
        self._window_created = False
        self._thread_exception = None

    def start(self):
        """Start the background thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread_exception = None
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="MediaPipeHandTracker"
        )
        self._thread.start()
        self._running = True

    def stop(self):
        """Signal the worker thread to stop. Call this from the main thread."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._running = False
        self.close_window()

    def close_window(self):
        """Destroy the OpenCV window from the main thread."""
        if not self._window_created:
            return
        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass
        self._window_created = False

    def is_running(self) -> bool:
        return self._running and not self._stop.is_set()
    
    def _raise_if_thread_failed(self):
        with self._lock:
            exc = self._thread_exception
        if exc is not None:
            raise RuntimeError("MediaPipeHandTracker worker thread crashed") from exc


    def get_mediapipe_angles(self, handeness="Right"):
        """
        Return the most recent set of angles.

        Order:
        ['Thumb_CMC', 'Thumb_MCP', 'Thumb_IP', 'Index_MCP', 'Index_PIP', 'Index_DIP',
        'Middle_MCP', 'Middle_PIP', 'Middle_DIP', 'Ring_MCP', 'Ring_PIP', 'Ring_DIP',
        'Pinky_MCP', 'Pinky_PIP', 'Pinky_DIP']
        """
        self._raise_if_thread_failed()

        with self._lock:
            if handeness == "Right":
                mediapipe_angles = deepcopy(self._latest_right_angles['angles'])
            else:
                mediapipe_angles = deepcopy(self._latest_left_angles['angles'])

        return [mediapipe_angles[k] for k in self.joint_names]
    
    def get_hand_state(self, handeness="Right"):
        """
        Return latest angles + landmarks from the same frame.

        Output dict:
            {
                'timestamp': float,
                'angles_list': [15 floats],
                'angles_dict': {...},
                'landmarks': np.ndarray shape (21, 3) or None,
                'landmarks_type': 'world' | 'normalized' | None
            }
        """
        self._raise_if_thread_failed()

        with self._lock:
            latest = self._latest_right_angles if handeness == "Right" else self._latest_left_angles

            timestamp = latest['timestamp']
            angles_dict = deepcopy(latest['angles'])
            landmarks = None if latest['landmarks'] is None else deepcopy(latest['landmarks'])
            landmarks_type = latest.get('landmarks_type', None)

        return {
            'timestamp': timestamp,
            'angles_list': [angles_dict[k] for k in self.joint_names],
            'angles_dict': angles_dict,
            'landmarks': landmarks,
            'landmarks_type': landmarks_type,
        }
        

    def get_last_frame(self):
        """Return a copy of the last annotated BGR frame (or None)."""
        self._raise_if_thread_failed()

        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def poll_gui(self) -> bool:
        """
        Show the latest frame from the MAIN thread only.

        Returns False when the user presses ESC or q.
        """
        self._raise_if_thread_failed()

        if not self.show_window:
            return True

        frame = self.get_last_frame()
        if frame is None:
            return True

        try:
            if not self._window_created:
                cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                self._window_created = True

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self.stop()
                return False
            return True
        except cv2.error as exc:
            raise RuntimeError(
                "OpenCV GUI failed. On macOS, cv2.namedWindow/imshow/waitKey must run on the main thread."
            ) from exc

    def _run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index, self.capture_backend)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Could not open camera index {self.camera_index} with backend {self.capture_backend}"
                )

            mp_hands = mp.solutions.hands
            mp_draw = mp.solutions.drawing_utils
            mp_styles = mp.solutions.drawing_styles

            with mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as hands:

                # IMPORTANT:
                # No cv2.namedWindow / cv2.imshow / cv2.waitKey here.
                # This worker thread only captures/processes frames.

                while not self._stop.is_set():
                    ok, frame_bgr = cap.read()
                    if not ok:
                        time.sleep(0.005)
                        continue

                    # Processing is always done on the non-mirrored frame
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    rgb.flags.writeable = False
                    results = hands.process(rgb)
                    rgb.flags.writeable = True

                    annotated = frame_bgr.copy()
                    h, w = annotated.shape[:2]

                    hands_out = []
                    if results.multi_hand_landmarks:
                        for i, hand_lms in enumerate(results.multi_hand_landmarks):
                            # Draw skeleton
                            mp_draw.draw_landmarks(
                                annotated,
                                hand_lms,
                                mp_hands.HAND_CONNECTIONS,
                                mp_styles.get_default_hand_landmarks_style(),
                                mp_styles.get_default_hand_connections_style(),
                            )

                            # Normalized image landmarks (always available)
                            normalized_pts = np.array(
                                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                                dtype=np.float64
                            )

                            # True world landmarks when available
                            if (
                                results.multi_hand_world_landmarks
                                and len(results.multi_hand_world_landmarks) > i
                            ):
                                world_pts = np.array(
                                    [[lm.x, lm.y, lm.z] for lm in results.multi_hand_world_landmarks[i].landmark],
                                    dtype=np.float64
                                )
                            else:
                                world_pts = None

                            # Keep the choice for angle computation
                            angle_pts = normalized_pts if self.use_2D_coord_for_angles else (
                                world_pts if world_pts is not None else normalized_pts
                            )

                            # For thumb pinch logic, prefer true world landmarks
                            stored_landmarks = world_pts if world_pts is not None else normalized_pts
                            stored_landmarks_type = 'world' if world_pts is not None else 'normalized'

                            # Angles from vectors
                            angles = self._compute_joint_angles(world_pts, mp_hands)

                            # 2D positions for labels
                            pts2d = [(int(p.x * w), int(p.y * h)) for p in hand_lms.landmark]
                            self._draw_angle_labels(annotated, pts2d, angles, mp_hands)

                            # Handedness (swap because processing image is not mirrored)
                            handed = None
                            if results.multi_handedness and len(results.multi_handedness) > i:
                                label = results.multi_handedness[i].classification[0].label
                                handed = ('Left' if label == 'Right' else 'Right')

                                wx, wy = pts2d[mp_hands.HandLandmark.WRIST.value]
                                cv2.putText(
                                    annotated,
                                    handed,
                                    (wx - 20, wy - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (255, 255, 255),
                                    2,
                                    cv2.LINE_AA
                                )

                            hands_out.append({
                                'handedness': handed,
                                'angles': angles,
                                'landmarks': stored_landmarks,
                                'landmarks_type': stored_landmarks_type,
                            })

                    display = cv2.flip(annotated, 1) if self.mirror else annotated

                    with self._lock:
                        for hand in hands_out:
                            payload = {
                                'timestamp': time.time(),
                                'angles': deepcopy(hand['angles']),
                                'landmarks': None if hand['landmarks'] is None else deepcopy(hand['landmarks']),
                                'landmarks_type': hand['landmarks_type'],
                            }

                            if hand['handedness'] == "Right":
                                self._latest_right_angles = payload
                            elif hand['handedness'] == "Left":
                                self._latest_left_angles = payload

                        self._latest_frame = display

        except Exception as exc:
            with self._lock:
                self._thread_exception = exc
            self._stop.set()

        finally:
            if cap is not None:
                cap.release()
            self._running = False


    @staticmethod
    def _angle_between(v1, v2, eps=1e-9):
        v1 = np.asarray(v1, dtype=np.float64)
        v2 = np.asarray(v2, dtype=np.float64)
        n1 = np.linalg.norm(v1) + eps
        n2 = np.linalg.norm(v2) + eps
        c = np.dot(v1, v2) / (n1 * n2)
        c = np.clip(c, -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    @classmethod
    def _angle_3pts(cls, a, b, c):
        # angle at b defined by vectors (a-b) and (c-b)
        return cls._angle_between(np.array(a) - np.array(b), np.array(c) - np.array(b))

    @staticmethod
    def _palm_center(world_pts, mp_hands):
        idx = [
            mp_hands.HandLandmark.WRIST.value,
            mp_hands.HandLandmark.THUMB_CMC.value,
            mp_hands.HandLandmark.INDEX_FINGER_MCP.value,
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP.value,
            mp_hands.HandLandmark.RING_FINGER_MCP.value,
            mp_hands.HandLandmark.PINKY_MCP.value
        ]
        return world_pts[idx].mean(axis=0)

    @classmethod
    def _compute_joint_angles(cls, world_pts, mp_hands):
        LM = mp_hands.HandLandmark
        ang = {}
        pc = cls._palm_center(world_pts, mp_hands)

        wrist = LM.WRIST.value
        th_cmc, th_mcp, th_ip, th_tip = LM.THUMB_CMC.value, LM.THUMB_MCP.value, LM.THUMB_IP.value, LM.THUMB_TIP.value
        ang['Thumb_CMC'] = cls._angle_3pts(world_pts[wrist], world_pts[th_cmc], world_pts[th_mcp])
        ang['Thumb_MCP'] = cls._angle_3pts(world_pts[th_cmc], world_pts[th_mcp], world_pts[th_ip])
        ang['Thumb_IP']  = cls._angle_3pts(world_pts[th_mcp], world_pts[th_ip], world_pts[th_tip])

        def finger(name, mcp):
            pip, dip, tip = mcp + 1, mcp + 2, mcp + 3
            ang[f'{name}_MCP'] = cls._angle_3pts(pc, world_pts[mcp], world_pts[pip])
            ang[f'{name}_PIP'] = cls._angle_3pts(world_pts[mcp], world_pts[pip], world_pts[dip])
            ang[f'{name}_DIP'] = cls._angle_3pts(world_pts[pip], world_pts[dip], world_pts[tip])

        finger('Index',  LM.INDEX_FINGER_MCP.value)
        finger('Middle', LM.MIDDLE_FINGER_MCP.value)
        finger('Ring',   LM.RING_FINGER_MCP.value)
        finger('Pinky',  LM.PINKY_MCP.value)
        return ang

    @staticmethod
    def _draw_angle_labels(img, lms2d, angles, mp_hands):
        def put(text, idx, dx=0, dy=-10):
            x, y = lms2d[idx]
            cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA)

        LM = mp_hands.HandLandmark
        put(f'CMC {angles["Thumb_CMC"]:.0f}', LM.THUMB_CMC.value)
        put(f'MCP {angles["Thumb_MCP"]:.0f}', LM.THUMB_MCP.value)
        put(f'IP  {angles["Thumb_IP"]:.0f}',  LM.THUMB_IP.value)

        for name, base in [('Index', LM.INDEX_FINGER_MCP.value),
                           ('Middle', LM.MIDDLE_FINGER_MCP.value),
                           ('Ring',   LM.RING_FINGER_MCP.value),
                           ('Pinky',  LM.PINKY_FINGER_MCP.value if hasattr(LM, 'PINKY_FINGER_MCP') else LM.PINKY_MCP.value)]:
            
            base = base
            put(f'{name} MCP {angles[f"{name}_MCP"]:.0f}', base)
            put(f'PIP {angles[f"{name}_PIP"]:.0f}', base + 1)
            put(f'DIP {angles[f"{name}_DIP"]:.0f}', base + 2)



if __name__ == "__main__":
    tracker = MediaPipeHandTracker(
        camera_index=0,
        mirror=False,
        show_window=True,   # okay now, because the main loop below calls poll_gui()
        use_2D_coord_for_angles=True
    )
    tracker.start()

    try:
        while True:
            right_hand_angles = tracker.get_mediapipe_angles()
            print(" ".join(
                f"{name}: {angle:.2f}"
                for name, angle in zip(tracker.joint_names, right_hand_angles)
            ))

            if tracker.show_window:
                if not tracker.poll_gui():
                    break

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        tracker.stop()