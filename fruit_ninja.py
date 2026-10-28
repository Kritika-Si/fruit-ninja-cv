"""
Fruit Ninja - Computer Vision Edition
Control and slice fruits using your finger through the webcam!
Powered by OpenCV and MediaPipe Hand Tracking.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random
import time
import math
import os
import threading

# Use winsound for low-latency native sound effects on Windows
try:
    import winsound
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False

def play_audio(filename):
    """Play sound asynchronously without blocking video loop."""
    if HAS_SOUND and os.path.exists(filename):
        def _play():
            try:
                winsound.PlaySound(filename, winsound.SND_ASYNC | winsound.SND_FILENAME)
            except Exception:
                pass
        threading.Thread(target=_play, daemon=True).start()

# --- Fruit Definitions ---
FRUIT_TYPES = {
    'watermelon': {
        'name': 'Watermelon',
        'outer_color': (34, 139, 34),     # Forest Green
        'inner_color': (60, 20, 220),     # Vibrant Red/Crimson (BGR)
        'flesh_color': (70, 70, 240),
        'radius': 48,
        'points': 10,
        'is_bomb': False
    },
    'orange': {
        'name': 'Orange',
        'outer_color': (0, 140, 255),     # Deep Orange (BGR)
        'inner_color': (0, 180, 255),     # Light Orange
        'flesh_color': (30, 200, 255),
        'radius': 42,
        'points': 10,
        'is_bomb': False
    },
    'apple': {
        'name': 'Apple',
        'outer_color': (30, 30, 220),     # Bright Red
        'inner_color': (180, 240, 240),   # Creamy white inside
        'flesh_color': (50, 50, 240),
        'radius': 40,
        'points': 10,
        'is_bomb': False
    },
    'banana': {
        'name': 'Banana',
        'outer_color': (20, 215, 245),    # Golden Yellow
        'inner_color': (190, 245, 250),   # Pale yellow
        'flesh_color': (40, 225, 255),
        'radius': 38,
        'points': 15,
        'is_bomb': False
    },
    'bomb': {
        'name': 'Bomb',
        'outer_color': (30, 30, 30),      # Dark metallic
        'inner_color': (10, 10, 10),
        'flesh_color': (0, 0, 255),
        'radius': 44,
        'points': 0,
        'is_bomb': True
    }
}

class Particle:
    def __init__(self, x, y, color, size=None, speed_mult=1.0):
        self.x = x
        self.y = y
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(3, 11) * speed_mult
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed - random.uniform(1, 4)
        self.color = color
        self.size = size if size else random.randint(3, 7)
        self.alpha = 1.0
        self.decay = random.uniform(0.025, 0.055)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.35  # gravity
        self.alpha -= self.decay
        self.size = max(1, self.size - 0.08)
        return self.alpha > 0

    def draw(self, img):
        if self.alpha <= 0:
            return
        cx, cy = int(self.x), int(self.y)
        h, w = img.shape[:2]
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(img, (cx, cy), max(1, int(self.size)), self.color, -1)

class SplatMark:
    def __init__(self, x, y, color):
        self.x = int(x)
        self.y = int(y)
        self.color = color
        self.radius = random.randint(30, 55)
        self.drops = []
        for _ in range(random.randint(6, 12)):
            ang = random.uniform(0, 2 * math.pi)
            dist = random.uniform(15, 60)
            r = random.randint(4, 10)
            self.drops.append((int(x + math.cos(ang) * dist), int(y + math.sin(ang) * dist), r))
        self.alpha = 0.55
        self.decay = 0.008

    def update(self):
        self.alpha -= self.decay
        return self.alpha > 0

    def draw(self, overlay):
        if self.alpha <= 0:
            return
        cv2.circle(overlay, (self.x, self.y), self.radius, self.color, -1)
        for dx, dy, r in self.drops:
            cv2.circle(overlay, (dx, dy), r, self.color, -1)

class SlicedHalf:
    def __init__(self, x, y, fruit_type, side, slice_angle):
        self.x = float(x)
        self.y = float(y)
        self.fruit_type = fruit_type
        self.side = side  # -1 for left, 1 for right
        self.angle = slice_angle
        # Push halves apart perpendicular to cut
        perp_angle = slice_angle + math.pi / 2
        spread = 5.0 * side
        self.vx = math.cos(perp_angle) * spread + random.uniform(-1.5, 1.5)
        self.vy = math.sin(perp_angle) * spread - random.uniform(2, 6)
        self.v_rot = random.uniform(4, 9) * side
        self.current_rot = math.degrees(slice_angle)
        self.alpha = 1.0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.55  # gravity
        self.current_rot += self.v_rot
        return self.y < 900  # offscreen check

    def draw(self, img):
        ft = FRUIT_TYPES[self.fruit_type]
        r = ft['radius']
        cx, cy = int(self.x), int(self.y)
        h, w = img.shape[:2]
        if not (-100 < cx < w + 100 and -100 < cy < h + 100):
            return

        rot_deg = self.current_rot
        start_a = rot_deg if self.side > 0 else (rot_deg + 180)
        end_a = start_a + 180

        # Draw outer rind half
        cv2.ellipse(img, (cx, cy), (r, r), 0, start_a, end_a, ft['outer_color'], -1)
        # Draw inner flesh half
        cv2.ellipse(img, (cx, cy), (int(r * 0.85), int(r * 0.85)), 0, start_a, end_a, ft['inner_color'], -1)
        # Flat cut edge highlight
        rad1 = math.radians(start_a)
        rad2 = math.radians(end_a)
        p1 = (int(cx + r * math.cos(rad1)), int(cy + r * math.sin(rad1)))
        p2 = (int(cx + r * math.cos(rad2)), int(cy + r * math.sin(rad2)))
        cv2.line(img, p1, p2, (255, 255, 255), 2)

class Fruit:
    def __init__(self, w, h, fruit_type=None):
        self.fruit_type = fruit_type if fruit_type else random.choice(
            ['watermelon', 'orange', 'apple', 'banana', 'bomb'] if random.random() < 0.22 else ['watermelon', 'orange', 'apple', 'banana']
        )
        self.info = FRUIT_TYPES[self.fruit_type]
        self.r = self.info['radius']
        self.x = random.randint(int(w * 0.15), int(w * 0.85))
        self.y = h + self.r + random.randint(5, 40)
        
        # Parabolic launch: aim slightly toward screen center
        center_bias = (w / 2 - self.x) * 0.007
        self.vx = random.uniform(-2.5, 2.5) + center_bias
        # Launch velocity tuned to peak between 15% and 40% screen height
        self.vy = -random.uniform(16.5, 21.0)
        self.gravity = 0.44
        self.rotation = random.uniform(0, 360)
        self.v_rot = random.uniform(-4, 4)
        self.sliced = False
        self.is_bomb = self.info['is_bomb']
        self.spark_timer = 0

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += self.gravity
        self.rotation += self.v_rot
        self.spark_timer += 1
        return self.y < 850

    def draw(self, img):
        if self.sliced:
            return
        cx, cy = int(self.x), int(self.y)
        r = self.r
        h, w = img.shape[:2]
        if not (-80 < cx < w + 80 and -80 < cy < h + 80):
            return

        if self.is_bomb:
            # Bomb body (dark sphere with specular highlight)
            cv2.circle(img, (cx, cy), r, (25, 25, 25), -1)
            cv2.circle(img, (cx - int(r*0.3), cy - int(r*0.3)), int(r*0.22), (90, 90, 90), -1)
            cv2.circle(img, (cx, cy), r, (60, 60, 60), 2)
            # Fuse
            fuse_start = (cx, cy - r)
            fuse_end = (cx + 12, cy - r - 16)
            cv2.line(img, fuse_start, fuse_end, (40, 90, 140), 3)
            # Sparking fuse tip
            spark_color = random.choice([(0, 255, 255), (0, 165, 255), (0, 0, 255), (255, 255, 255)])
            cv2.circle(img, fuse_end, random.randint(4, 8), spark_color, -1)
            # Hazard Skull / 'X' Mark on Bomb
            cv2.line(img, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 220), 3)
            cv2.line(img, (cx + 10, cy - 10), (cx - 10, cy + 10), (0, 0, 220), 3)
        else:
            # Watermelon
            if self.fruit_type == 'watermelon':
                cv2.circle(img, (cx, cy), r, self.info['outer_color'], -1)
                # Outer stripes
                for ang in range(0, 360, 45):
                    rad = math.radians(ang + self.rotation)
                    p_end = (int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad)))
                    cv2.line(img, (cx, cy), p_end, (15, 80, 20), 3)
                cv2.circle(img, (cx - int(r*0.3), cy - int(r*0.3)), int(r*0.2), (60, 180, 60), -1)
                cv2.circle(img, (cx, cy), r, (15, 80, 20), 2)
            # Apple
            elif self.fruit_type == 'apple':
                cv2.circle(img, (cx, cy), r, self.info['outer_color'], -1)
                cv2.circle(img, (cx - int(r*0.3), cy - int(r*0.3)), int(r*0.2), (100, 100, 255), -1)
                # Stem & leaf
                cv2.line(img, (cx, cy - r), (cx + 4, cy - r - 12), (30, 70, 110), 3)
                cv2.ellipse(img, (cx + 8, cy - r - 8), (7, 4), 30, 0, 360, (40, 180, 40), -1)
            # Orange
            elif self.fruit_type == 'orange':
                cv2.circle(img, (cx, cy), r, self.info['outer_color'], -1)
                cv2.circle(img, (cx - int(r*0.3), cy - int(r*0.3)), int(r*0.2), (100, 200, 255), -1)
                cv2.circle(img, (cx, cy), 4, (0, 100, 180), -1)
            # Banana
            elif self.fruit_type == 'banana':
                cv2.ellipse(img, (cx, cy), (r + 10, int(r * 0.6)), int(self.rotation), 0, 360, self.info['outer_color'], -1)
                cv2.ellipse(img, (cx, cy), (r + 6, int(r * 0.45)), int(self.rotation), 0, 360, (80, 240, 255), -1)
                # Brown tips
                rad = math.radians(self.rotation)
                tip1 = (int(cx + (r + 10) * math.cos(rad)), int(cy + (r + 10) * math.sin(rad)))
                cv2.circle(img, tip1, 4, (30, 60, 100), -1)

def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Calculates minimum distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return math.hypot(px - closest_x, py - closest_y)

class FruitNinjaGame:
    def __init__(self, cam_index=0):
        # Setup OpenCV camera
        self.cam_index = cam_index
        self.cap = cv2.VideoCapture(self.cam_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Read a frame to get actual dimensions
        ret, frame = self.cap.read()
        if not ret:
            print(f"WARNING: Camera index {cam_index} not accessible. Trying camera index 1...")
            self.cap.release()
            self.cam_index = 1
            self.cap = cv2.VideoCapture(self.cam_index)
            ret, frame = self.cap.read()
            if not ret:
                print("WARNING: Could not access camera. Defaulting to 1280x720 window.")
                self.w, self.h = 1280, 720
            else:
                self.h, self.w = frame.shape[:2]
        else:
            self.h, self.w = frame.shape[:2]

        self.cam_toast = f"Active Camera: {self.cam_index}"
        self.toast_timer = 60

        # Setup MediaPipe HandLandmarker
        model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found!")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # Game state
        self.state = 'START'  # 'START', 'PLAYING', 'GAMEOVER'
        self.score = 0
        self.high_score = 0
        self.lives = 3
        self.fruits = []
        self.sliced_halves = []
        self.particles = []
        self.splats = []
        
        # Blade trail tracking
        self.trail = []  # stores (x, y, timestamp)
        self.max_trail_len = 16
        
        # Combo tracker
        self.recent_slices = []  # timestamps of slices
        self.combo_text = ""
        self.combo_timer = 0
        self.combo_pos = (self.w // 2, self.h // 2)

        # Spawn timing
        self.last_spawn_time = time.time()
        self.spawn_interval = 2.0

        # Start button fruit
        self.start_fruit = Fruit(self.w, self.h, 'watermelon')
        self.start_fruit.x = self.w // 2
        self.start_fruit.y = self.h // 2 + 30
        self.start_fruit.vx = 0
        self.start_fruit.vy = 0
        self.start_fruit.gravity = 0

    def switch_camera(self):
        next_index = (self.cam_index + 1) % 3
        print(f"Switching camera from {self.cam_index} to {next_index}...")
        self.cap.release()
        new_cap = cv2.VideoCapture(next_index)
        ret, frame = new_cap.read()
        if ret:
            self.cam_index = next_index
            self.cap = new_cap
            self.h, self.w = frame.shape[:2]
            self.cam_toast = f"Switched to Camera {self.cam_index}"
            self.toast_timer = 70
            print(f"Camera successfully switched to index {self.cam_index}!")
        else:
            print(f"Camera index {next_index} not accessible. Reverting to index {self.cam_index}.")
            new_cap.release()
            self.cap = cv2.VideoCapture(self.cam_index)
            self.cam_toast = f"Camera {next_index} unavailable"
            self.toast_timer = 50

    def reset_game(self):
        self.state = 'PLAYING'
        self.score = 0
        self.lives = 3
        self.fruits.clear()
        self.sliced_halves.clear()
        self.particles.clear()
        self.splats.clear()
        self.trail.clear()
        self.recent_slices.clear()
        self.combo_text = ""
        self.last_spawn_time = time.time()
        self.spawn_interval = 2.0
        play_audio('slice.wav')

    def spawn_wave(self):
        # Spawn 1 to 3 fruits in a wave
        now = time.time()
        if now - self.last_spawn_time > self.spawn_interval:
            self.last_spawn_time = now
            # Gradually speed up spawn rate as score grows
            self.spawn_interval = max(1.1, 2.3 - (self.score * 0.015))
            
            num_fruits = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
            for _ in range(num_fruits):
                self.fruits.append(Fruit(self.w, self.h))

    def check_slash(self, p1, p2):
        """Check if blade movement from p1 to p2 slices any fruit."""
        slash_angle = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        sliced_in_this_stroke = 0

        # Check in START screen
        if self.state == 'START':
            dist = point_to_segment_distance(self.start_fruit.x, self.start_fruit.y, p1[0], p1[1], p2[0], p2[1])
            if dist <= self.start_fruit.r:
                self.trigger_slice(self.start_fruit, slash_angle)
                self.reset_game()
            return

        # Check in GAMEOVER screen
        if self.state == 'GAMEOVER':
            dist = point_to_segment_distance(self.start_fruit.x, self.start_fruit.y, p1[0], p1[1], p2[0], p2[1])
            if dist <= self.start_fruit.r:
                self.trigger_slice(self.start_fruit, slash_angle)
                self.reset_game()
            return

        # Check active game fruits
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            dist = point_to_segment_distance(fruit.x, fruit.y, p1[0], p1[1], p2[0], p2[1])
            if dist <= fruit.r:
                self.trigger_slice(fruit, slash_angle)
                sliced_in_this_stroke += 1

        if sliced_in_this_stroke > 0:
            now = time.time()
            self.recent_slices.append(now)
            # Combo check: slices within 0.35s
            self.recent_slices = [t for t in self.recent_slices if now - t < 0.35]
            combo_count = len(self.recent_slices)
            if combo_count >= 2:
                bonus = combo_count * 10
                self.score += bonus
                self.combo_text = f"COMBO x{combo_count}! +{bonus}"
                self.combo_timer = 30
                self.combo_pos = (int(p2[0]), max(60, int(p2[1]) - 40))
                play_audio('combo.wav')

    def trigger_slice(self, fruit, slice_angle):
        fruit.sliced = True
        
        if fruit.is_bomb:
            play_audio('bomb.wav')
            # Massive explosion particles
            for _ in range(70):
                self.particles.append(Particle(fruit.x, fruit.y, (0, 0, 255), speed_mult=1.8))
                self.particles.append(Particle(fruit.x, fruit.y, (0, 165, 255), speed_mult=1.5))
                self.particles.append(Particle(fruit.x, fruit.y, (200, 200, 200), speed_mult=1.2))
            # Direct Game Over
            self.lives = 0
            self.state = 'GAMEOVER'
            if self.score > self.high_score:
                self.high_score = self.score
            play_audio('gameover.wav')
            # Recreate start fruit for restart
            self.start_fruit = Fruit(self.w, self.h, 'watermelon')
            self.start_fruit.x = self.w // 2
            self.start_fruit.y = self.h // 2 + 100
            self.start_fruit.vx = 0
            self.start_fruit.vy = 0
            self.start_fruit.gravity = 0
            return

        # Sliced a normal fruit!
        play_audio('slice.wav')
        self.score += fruit.info['points']
        if self.score > self.high_score:
            self.high_score = self.score

        # Add sliced halves
        self.sliced_halves.append(SlicedHalf(fruit.x, fruit.y, fruit.fruit_type, -1, slice_angle))
        self.sliced_halves.append(SlicedHalf(fruit.x, fruit.y, fruit.fruit_type, 1, slice_angle))

        # Add juice splat on screen
        self.splats.append(SplatMark(fruit.x, fruit.y, fruit.info['inner_color']))

        # Add juice droplets particles
        for _ in range(28):
            self.particles.append(Particle(fruit.x, fruit.y, fruit.info['flesh_color']))
            self.particles.append(Particle(fruit.x, fruit.y, fruit.info['outer_color'], size=4))

    def update_and_render(self, frame):
        # 1. Flip camera frame horizontally for intuitive mirror interaction
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        # 2. Hand tracking via MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.detector.detect(mp_image)

        # Detect index finger tip (Landmark 8)
        current_finger_pos = None
        if detection_result.hand_landmarks:
            for hand_landmarks in detection_result.hand_landmarks:
                tip = hand_landmarks[8]
                fx, fy = int(tip.x * w), int(tip.y * h)
                current_finger_pos = (fx, fy)
                break  # Track primary hand

        # Update blade trail
        now = time.time()
        if current_finger_pos:
            if len(self.trail) > 0:
                p_prev = self.trail[-1][:2]
                # Check for slash collision
                self.check_slash(p_prev, current_finger_pos)
            self.trail.append((current_finger_pos[0], current_finger_pos[1], now))
        
        # Remove old trail points (keep last 0.22s)
        self.trail = [p for p in self.trail if now - p[2] < 0.22]

        # 3. Render Splatters (semi-transparent juice stains)
        if len(self.splats) > 0:
            splat_overlay = np.zeros_like(frame)
            for splat in self.splats[:]:
                if splat.update():
                    splat.draw(splat_overlay)
                else:
                    self.splats.remove(splat)
            # Alpha blend splats
            cv2.addWeighted(splat_overlay, 0.45, frame, 1.0, 0, frame)

        # 4. State: PLAYING logic
        if self.state == 'PLAYING':
            self.spawn_wave()
            
            # Update fruits
            for fruit in self.fruits[:]:
                is_alive = fruit.update()
                if not is_alive:
                    # Missed a fruit!
                    if not fruit.sliced and not fruit.is_bomb:
                        self.lives -= 1
                        play_audio('splat.wav')
                        if self.lives <= 0:
                            self.state = 'GAMEOVER'
                            if self.score > self.high_score:
                                self.high_score = self.score
                            play_audio('gameover.wav')
                            # Prepare restart button
                            self.start_fruit = Fruit(self.w, self.h, 'watermelon')
                            self.start_fruit.x = self.w // 2
                            self.start_fruit.y = self.h // 2 + 100
                            self.start_fruit.vx = 0
                            self.start_fruit.vy = 0
                            self.start_fruit.gravity = 0
                    self.fruits.remove(fruit)
                else:
                    fruit.draw(frame)

        # Draw Sliced Halves
        for half in self.sliced_halves[:]:
            if half.update():
                half.draw(frame)
            else:
                self.sliced_halves.remove(half)

        # Draw Particles
        for particle in self.particles[:]:
            if particle.update():
                particle.draw(frame)
            else:
                self.particles.remove(particle)

        # 5. Draw Glowing Blade Trail
        if len(self.trail) > 1:
            for i in range(1, len(self.trail)):
                pt1 = self.trail[i - 1][:2]
                pt2 = self.trail[i][:2]
                ratio = i / len(self.trail)
                
                # Outer cyan/golden glow
                glow_thick = max(2, int(18 * ratio))
                cv2.line(frame, pt1, pt2, (255, 220, 50), glow_thick)
                # Inner sharp white katana edge
                core_thick = max(1, int(6 * ratio))
                cv2.line(frame, pt1, pt2, (255, 255, 255), core_thick)

        # Finger tip sparkle
        if current_finger_pos:
            cv2.circle(frame, current_finger_pos, 8, (255, 255, 255), -1)
            cv2.circle(frame, current_finger_pos, 14, (0, 255, 255), 2)

        # 6. Draw HUD & UI Overlays
        self.draw_hud(frame)

        return frame

    def draw_hud(self, img):
        h, w = img.shape[:2]

        if self.state == 'START':
            # Semi-transparent dark overlay for banner
            overlay = img.copy()
            cv2.rectangle(overlay, (w//2 - 380, 80), (w//2 + 380, 220), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
            
            # Title
            cv2.putText(img, "FRUIT NINJA CV", (w//2 - 270, 150), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 215, 255), 4)
            cv2.putText(img, "Slice the Watermelon with your finger to PLAY!", (w//2 - 320, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            
            # Start fruit
            self.start_fruit.draw(img)
            cv2.putText(img, "SLICE ME!", (self.start_fruit.x - 70, self.start_fruit.y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 255, 50), 2)

            # Help instructions at bottom
            cv2.putText(img, "Show your index finger to the camera and swipe across fruits to slice!", (40, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 240, 255), 2)

        elif self.state == 'PLAYING':
            # Camera notification toast
            if self.toast_timer > 0:
                self.toast_timer -= 1
                cv2.rectangle(img, (w//2 - 170, 20), (w//2 + 170, 60), (30, 30, 30), -1)
                cv2.rectangle(img, (w//2 - 170, 20), (w//2 + 170, 60), (0, 215, 255), 2)
                cv2.putText(img, self.cam_toast, (w//2 - 140, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.rectangle(img, (25, 20), (230, 85), (0, 0, 0), -1)
            cv2.rectangle(img, (25, 20), (230, 85), (0, 180, 255), 2)
            cv2.putText(img, f"SCORE: {self.score}", (40, 65), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2)

            # High Score
            cv2.putText(img, f"BEST: {self.high_score}", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 215, 255), 2)

            # Lives / Hearts (Top Right)
            start_x = w - 160
            for i in range(3):
                lx = start_x + i * 45
                ly = 50
                if i < self.lives:
                    # Red Heart
                    cv2.circle(img, (lx - 6, ly), 9, (0, 0, 240), -1)
                    cv2.circle(img, (lx + 6, ly), 9, (0, 0, 240), -1)
                    triangle = np.array([[lx - 14, ly + 2], [lx + 14, ly + 2], [lx, ly + 18]], np.int32)
                    cv2.fillPoly(img, [triangle], (0, 0, 240))
                else:
                    # Lost life 'X'
                    cv2.line(img, (lx - 10, ly - 10), (lx + 10, ly + 10), (80, 80, 80), 3)
                    cv2.line(img, (lx + 10, ly - 10), (lx - 10, ly + 10), (80, 80, 80), 3)

            # Combo Banner
            if self.combo_timer > 0:
                self.combo_timer -= 1
                cx, cy = self.combo_pos
                cv2.putText(img, self.combo_text, (cx - 80, cy), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 215, 255), 3)
                cv2.putText(img, self.combo_text, (cx - 80, cy), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 1)

        elif self.state == 'GAMEOVER':
            overlay = img.copy()
            cv2.rectangle(overlay, (w//2 - 320, 80), (w//2 + 320, 320), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

            cv2.putText(img, "GAME OVER", (w//2 - 180, 150), cv2.FONT_HERSHEY_DUPLEX, 1.8, (0, 0, 255), 4)
            cv2.putText(img, f"Final Score: {self.score}", (w//2 - 120, 215), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(img, f"High Score:  {self.high_score}", (w//2 - 120, 260), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 215, 255), 2)
            cv2.putText(img, "Press [R] or Slice Fruit Below to Restart", (w//2 - 240, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 240, 255), 2)

            # Restart fruit
            self.start_fruit.draw(img)
            cv2.putText(img, "RESTART!", (self.start_fruit.x - 70, self.start_fruit.y + 80), cv2.FONT_HERSHEY_DUPLEX, 0.9, (50, 255, 50), 2)

    def run(self):
        print("==================================================")
        print("  FRUIT NINJA COMPUTER VISION - GAME STARTED!     ")
        print("  Controls:                                       ")
        print("  - Point your Index Finger to slice fruits!      ")
        print("  - Slice 2+ fruits quickly for COMBO bonus!      ")
        print("  - Avoid BOMB! Slicing a bomb ends the game!     ")
        print("  - Don't let fruits fall! (3 Lives)              ")
        print("  - Press [C] to Switch Camera (Laptop/External)  ")
        print("  - Press [R] to Restart | Press [Q] to Quit      ")
        print("==================================================")

        window_name = "Fruit Ninja - Computer Vision"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.w, self.h)

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            rendered_frame = self.update_and_render(frame)
            cv2.imshow(window_name, rendered_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('r'):
                self.reset_game()
            elif key == ord('c'):
                self.switch_camera()

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    game = FruitNinjaGame()
    game.run()
