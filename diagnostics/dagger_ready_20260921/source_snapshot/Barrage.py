import argparse
import csv
import json
import math
import multiprocessing as mp
import tkinter
import time
from pathlib import Path

import numpy as np
import pygame

from barrage_rl.causal_control import CausalActionPipeline
from barrage_rl.timing import DEFAULT_ACTION_REPEAT, PHYSICS_FPS as FIXED_PHYSICS_FPS
from barrage_rl.runtime_core import (
    ACTION_VECTORS,
    BulletFieldConfig,
    OPENING_BATCH_COUNT as CORE_OPENING_BATCH_COUNT,
    OPENING_BATCH_INTERVAL_SECONDS as CORE_OPENING_BATCH_INTERVAL_SECONDS,
    advance_bullet_field,
    advance_plane,
    colliding_bullet_indices,
    normalized_direction,
    opening_batch_physics_step,
    opening_batch_size,
    render_world_surface,
    spawn_bullets,
)
from barrage_rl.task_spec import (
    PRODUCTION_ANALYTIC_SHIELD,
    PRODUCTION_ANALYTIC_SHIELD_GATE,
    TARGET_TASK,
)


FIXED_SCREEN_WIDTH = 820
FIXED_SCREEN_HEIGHT = 820
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AI_CHECKPOINT = (
    PROJECT_ROOT
    / "diagnostics"
    / "risk_removal_20260920"
    / "policy_teacher_cost.pt"
)


def _latency_statistics(values):
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean_ms": 0.0, "median_ms": 0.0,
                "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    return {
        "count": int(array.size),
        "mean_ms": float(array.mean()),
        "median_ms": float(np.median(array)),
        "p95_ms": float(np.percentile(array, 95.0)),
        "p99_ms": float(np.percentile(array, 99.0)),
        "max_ms": float(array.max()),
    }


def _collision_state_text():
    """Return the label that describes the current wall-collision state."""
    return "开启" if Barrage.COLLISION else "关闭"


def _start_music_async(path):
    """Initialize optional audio without blocking the window event loop."""
    process = mp.get_context("spawn").Process(
        target=_run_music_worker,
        args=(path,),
        name="barrage-music-init",
        daemon=True,
    )
    process.start()
    return process


def _run_music_worker(path):
    music = Music(path)
    while music.started and pygame.mixer.music.get_busy():
        pygame.time.wait(250)


def settings():
    entry_dic = {}
    should_start = False

    def is_digit_input(value):
        return not value or value.isdecimal()

    def button_c():
        Barrage.COLLISION = not Barrage.COLLISION
        button_collision.config(text=_collision_state_text())

    def button_i():
        Barrage.INVINCIBLE = not Barrage.INVINCIBLE
        if Barrage.INVINCIBLE:
            button_invincible.config(text="开启")
        else:
            button_invincible.config(text="关闭")

    def button_m():
        Barrage.MUSIC = not Barrage.MUSIC
        if Barrage.MUSIC:
            button_music.config(image=photo_on)
        else:
            button_music.config(image=photo_off)

    def close():
        nonlocal should_start
        bullet_size_text = entry_dic["bullet_size"].get()
        quantity_text = entry_dic["quantity"].get()
        if not bullet_size_text or not quantity_text:
            return

        bullet_size = int(bullet_size_text)
        quantity = int(quantity_text)
        if not 1 <= bullet_size <= 10 or quantity < 1:
            return

        Barrage.SCREEN_WIDTH = FIXED_SCREEN_WIDTH
        Barrage.SCREEN_HEIGHT = FIXED_SCREEN_HEIGHT
        Barrage.BULLET_SIZE = bullet_size
        Barrage.QUANTITY = quantity
        Barrage.TimeSize = int(math.sqrt(Barrage.SCREEN_WIDTH * Barrage.SCREEN_HEIGHT) / 25)
        should_start = True
        root.destroy()

    root = tkinter.Tk()
    width, height = root.maxsize()
    root.geometry(f"{230}x{283}+{int(width / 2.4)}+{int(height / 3)}")
    root.title("Setting")
    root.resizable(False, False)
    validate_digits = (root.register(is_digit_input), "%P")
    tkinter.Label(root, text="窗口宽度:", font="宋体 20").grid(row=0, column=0)
    tkinter.Label(root, text="窗口高度:", font="宋体 20").grid(row=1, column=0)
    tkinter.Label(root, text="子弹大小:", font="宋体 20").grid(row=2, column=0)
    tkinter.Label(root, text="子弹数量:", font="宋体 20").grid(row=3, column=0)
    tkinter.Label(root, text="墙体碰撞:", font="宋体 20").grid(row=4, column=0)
    tkinter.Label(root, text="子弹伤害:", font="宋体 20").grid(row=5, column=0)
    for row, value in enumerate((FIXED_SCREEN_WIDTH, FIXED_SCREEN_HEIGHT)):
        tkinter.Label(
            root,
            text=str(value),
            width=8,
            font="timesnewroman 22",
            anchor="center",
        ).grid(row=row, column=1, columnspan=2)
    for row, field, value in (
        (2, "bullet_size", "5"),
        (3, "quantity", str(TARGET_TASK.bullet_count)),
    ):
        widget = tkinter.Entry(
            root,
            width=8,
            font="timesnewroman 22",
            justify="center",
            validate="key",
            validatecommand=validate_digits,
        )
        widget.grid(row=row, column=1, columnspan=2)
        widget.insert(0, value)
        entry_dic[field] = widget
    button_collision = tkinter.Button(
        root,
        text=_collision_state_text(),
        command=button_c,
        width=6,
        height=0,
        font="宋体 22",
    )
    button_collision.grid(row=4, column=1)
    button_invincible = tkinter.Button(root, text="开启", command=button_i, width=6, height=0, font="宋体 22")
    button_invincible.grid(row=5, column=1)

    photo_on = tkinter.PhotoImage(
        file=str(PROJECT_ROOT / "image" / "music_on.png")
    )
    photo_off = tkinter.PhotoImage(
        file=str(PROJECT_ROOT / "image" / "music_off.png")
    )
    button_music = tkinter.Button(root, command=button_m, image=photo_on)
    button_music.grid(row=6, column=0)

    button_start = tkinter.Button(root, text="PLAY", command=close, width=6, height=0, font="timesnewroman  22")
    button_start.grid(row=6, column=1)
    root.mainloop()
    return should_start


class Barrage:
    window = None
    PLANE = None
    BULLET = None
    KEY = True
    COLLISION = False
    INVINCIBLE = True
    MUSIC = True
    SCREEN_WIDTH = FIXED_SCREEN_WIDTH
    SCREEN_HEIGHT = FIXED_SCREEN_HEIGHT
    BULLET_SIZE = 5
    QUANTITY = TARGET_TASK.bullet_count
    TimeStart = int()
    TimeNow = int()
    ALIVE_PHYSICS_STEPS = 0
    TimeSize = int()
    TimeColor = str()
    TimeFont = None
    TimeText = None
    TimeTextKey = None
    RestartText = None
    FPS = 100
    PHYSICS_FPS = FIXED_PHYSICS_FPS
    MAX_FRAME_TIME = 0.25
    PLANE_SPEED = 240.0
    BULLET_SPEED = 240.0
    TARGETED_BULLET_PROBABILITY = 0.10
    TARGETED_PREDICTION_SCALE_MIN = 0.65
    TARGETED_PREDICTION_SCALE_MAX = 1.0
    TARGETED_ANGULAR_NOISE = 0.08
    OPENING_BATCH_COUNT = CORE_OPENING_BATCH_COUNT
    OPENING_BATCH_INTERVAL_SECONDS = CORE_OPENING_BATCH_INTERVAL_SECONDS
    OPENING_SPAWNED_BATCHES = 0
    OPENING_EFFECTIVE_PHASE_SECONDS = 0.0
    RNG = np.random.default_rng()
    AI_CONTROLLER = None
    AI_PIPELINE = None
    AI_ACTION = 0
    LATENCY_TEST_SECONDS = 0.0
    LATENCY_OUTPUT = None
    BULLET_FIELD_CONFIG = None
    PRESSED_SCANCODES = set()

    @staticmethod
    def start_game():
        pygame.display.init()
        pygame.font.init()
        Barrage.window = pygame.display.set_mode((Barrage.SCREEN_WIDTH, Barrage.SCREEN_HEIGHT))
        pygame.display.set_caption("Barrage")
        pygame.key.stop_text_input()
        Barrage.PRESSED_SCANCODES.clear()

        if Barrage.AI_CONTROLLER is not None:
            Barrage.AI_PIPELINE = (
                CausalActionPipeline(
                    Barrage.AI_CONTROLLER,
                    decision_period_seconds=Barrage.AI_CONTROLLER.decision_interval / Barrage.PHYSICS_FPS,
                )
                if Barrage.AI_CONTROLLER.action_delay_steps == 1
                else None
            )

        # 字体只创建一次，避免在游戏循环中反复搜索和创建系统字体
        Barrage.TimeFont = pygame.font.SysFont("timesnewroman", Barrage.TimeSize)
        restart_font = pygame.font.SysFont("timesnewroman", int(0.7 * Barrage.TimeSize))
        Barrage.RestartText = restart_font.render(
            "PRESS SPACE OR ENTER TO RESTART", True, "#CD7F32"
        )
        Barrage.reset_game()
        physics_steps_total = 0
        if Barrage.AI_PIPELINE is not None:
            Barrage.render_world()
            Barrage.AI_PIPELINE.submit_surface(
                Barrage.window,
                capture_boundary=0,
                apply_boundary=1,
                enforce_deadline=False,
            )
            Barrage.AI_PIPELINE.wait_until_ready(1)
            Barrage.AI_PIPELINE.begin_measurement()

        if Barrage.MUSIC:
            _start_music_async(
                PROJECT_ROOT
                / "music"
                / "坂元信也,寺島里恵,前沢秀憲 - Starfield (ステージ2 BGM) - 沙羅曼蛇 (FC版).mp3"
            )

        # 渲染上限100 FPS，物理始终使用固定120 Hz子步。
        clock = pygame.time.Clock()
        physics_step = 1.0 / Barrage.PHYSICS_FPS
        accumulator = 0.0
        latency_enabled = Barrage.LATENCY_TEST_SECONDS > 0.0
        latency_started = time.perf_counter() if latency_enabled else 0.0
        previous_frame_started = None
        frame_intervals = []
        frame_processing = []
        physics_steps_per_frame = []
        synchronous_decision_ms = []
        frame_samples = []

        while True:
            frame_started = time.perf_counter() if latency_enabled else 0.0
            if latency_enabled:
                if previous_frame_started is not None:
                    frame_intervals.append(
                        (frame_started - previous_frame_started) * 1000.0
                    )
                previous_frame_started = frame_started
            frame_time = min(
                clock.tick(Barrage.FPS) / 1000.0, Barrage.MAX_FRAME_TIME
            )
            switch_skin, direction, restarted = Barrage.get_event()
            if restarted:
                accumulator = 0.0
                physics_steps_total = 0
                if Barrage.AI_PIPELINE is not None:
                    Barrage.render_world()
                    Barrage.AI_PIPELINE.submit_surface(
                        Barrage.window,
                        capture_boundary=0,
                        apply_boundary=1,
                        enforce_deadline=False,
                    )
                    Barrage.AI_PIPELINE.wait_until_ready(1)
                    Barrage.AI_PIPELINE.begin_measurement()
            if switch_skin:
                Barrage.PLANE.change_skin()

            accumulator += frame_time
            physics_steps = 0
            world_rendered = False
            while accumulator >= physics_step:
                step_direction = (
                    ACTION_VECTORS[Barrage.AI_ACTION]
                    if Barrage.AI_CONTROLLER is not None
                    else direction
                )
                Barrage.advance_physics(step_direction, physics_step)
                world_rendered = False
                accumulator -= physics_step
                physics_steps += 1
                physics_steps_total += 1
                if (
                    Barrage.AI_CONTROLLER is not None
                    and physics_steps_total
                    % Barrage.AI_CONTROLLER.decision_interval
                    == 0
                ):
                    boundary = (
                        physics_steps_total
                        // Barrage.AI_CONTROLLER.decision_interval
                    )
                    Barrage.render_world()
                    world_rendered = True
                    if Barrage.AI_PIPELINE is not None:
                        Barrage.AI_ACTION = Barrage.AI_PIPELINE.action_for_boundary(
                            boundary, Barrage.AI_ACTION
                        )
                        Barrage.AI_PIPELINE.submit_surface(
                            Barrage.window,
                            capture_boundary=boundary,
                            apply_boundary=boundary + 1,
                        )
                    else:
                        decision_started = time.perf_counter()
                        Barrage.AI_ACTION = (
                            Barrage.AI_CONTROLLER.observe_due_surface(Barrage.window)
                        )
                        if latency_enabled:
                            synchronous_decision_ms.append(
                                (time.perf_counter() - decision_started) * 1000.0
                            )
            if latency_enabled:
                physics_steps_per_frame.append(physics_steps)

            if not world_rendered:
                Barrage.render_world()
            Barrage.get_time()
            pygame.display.flip()
            if latency_enabled:
                frame_finished = time.perf_counter()
                frame_processing.append(
                    (frame_finished - frame_started) * 1000.0
                )
                frame_samples.append((
                    frame_finished - latency_started,
                    frame_processing[-1],
                    physics_steps,
                    int(Barrage.KEY),
                ))

            elapsed_seconds = (
                time.perf_counter() - latency_started if latency_enabled else 0.0
            )
            if latency_enabled and elapsed_seconds >= Barrage.LATENCY_TEST_SECONDS:
                decision_interval_steps = (
                    Barrage.AI_CONTROLLER.decision_interval
                    if Barrage.AI_CONTROLLER is not None else DEFAULT_ACTION_REPEAT
                )
                decision_budget_ms = 1000.0 * decision_interval_steps / Barrage.PHYSICS_FPS
                pipeline_report = (
                    Barrage.AI_PIPELINE.report()
                    if Barrage.AI_PIPELINE is not None
                    else {}
                )
                inference = (
                    pipeline_report.get("inference", _latency_statistics([]))
                    if Barrage.AI_PIPELINE is not None
                    else _latency_statistics(synchronous_decision_ms)
                )
                decision_over_budget_count = 0
                if inference["count"]:
                    # The aggregate does not retain samples; actual causal
                    # deadline misses are reported separately below.
                    decision_over_budget_count = (
                        int(pipeline_report.get("deadline_miss_count", 0))
                        if Barrage.AI_PIPELINE is not None
                        else int(np.count_nonzero(
                            np.asarray(synchronous_decision_ms) > decision_budget_ms
                        ))
                    )
                report = {
                    "mode": (
                        "visible_pygame_window_causal_pipeline"
                        if Barrage.AI_PIPELINE is not None
                        else ("visible_pygame_window_synchronous_legacy"
                              if Barrage.AI_CONTROLLER is not None
                              else "visible_pygame_window_no_ai")
                    ),
                    "observation_source": "Barrage.py rendered RGB surface",
                    "ai_enabled": Barrage.AI_CONTROLLER is not None,
                    "damage_immunity": not Barrage.INVINCIBLE,
                    "duration_seconds": float(elapsed_seconds),
                    "rendered_frames": int(len(frame_processing)),
                    "rendered_fps": float(len(frame_processing) / elapsed_seconds),
                    "render_fps_cap": int(Barrage.FPS),
                    "physics_fps": int(Barrage.PHYSICS_FPS),
                    "decision_hz_target": Barrage.PHYSICS_FPS / decision_interval_steps,
                    "decision_budget_ms": decision_budget_ms,
                    "bullets": int(Barrage.QUANTITY),
                    "targeted_bullet_probability": float(
                        Barrage.TARGETED_BULLET_PROBABILITY
                    ),
                    "frame_interval": _latency_statistics(frame_intervals),
                    "frame_processing": _latency_statistics(frame_processing),
                    "ai_all_observations": inference,
                    "ai_decision": inference,
                    "ai_nondecision": _latency_statistics([]),
                    "decision_interval": pipeline_report.get(
                        "submission_interval", _latency_statistics([])
                    ),
                    "decision_over_budget_count": decision_over_budget_count,
                    "decision_over_budget_fraction": float(
                        pipeline_report.get("deadline_miss_fraction", 0.0)
                        if Barrage.AI_PIPELINE is not None
                        else decision_over_budget_count / max(len(synchronous_decision_ms), 1)
                    ),
                    "causal_pipeline": pipeline_report,
                    "physics_catchup_frames": int(
                        np.count_nonzero(
                            np.asarray(physics_steps_per_frame, dtype=np.int32) > 1
                        )
                    ),
                    "physics_max_steps_per_frame": int(
                        max(physics_steps_per_frame, default=0)
                    ),
                }
                text = json.dumps(report, indent=2)
                print(text, flush=True)
                if Barrage.LATENCY_OUTPUT is not None:
                    output = Path(Barrage.LATENCY_OUTPUT)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(output.suffix + ".tmp")
                    temporary.write_text(text, encoding="utf-8")
                    temporary.replace(output)
                    samples_output = output.with_suffix(".frames.csv")
                    samples_temporary = samples_output.with_suffix(".csv.tmp")
                    with samples_temporary.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.writer(stream)
                        writer.writerow(("elapsed_seconds", "frame_processing_ms",
                                         "physics_steps", "alive"))
                        writer.writerows(frame_samples)
                    samples_temporary.replace(samples_output)
                if Barrage.AI_PIPELINE is not None:
                    Barrage.AI_PIPELINE.close()
                    Barrage.AI_PIPELINE = None
                pygame.quit()
                return report

    @staticmethod
    def reset_game():
        """重置一局游戏，避免重新进入 start_game 造成循环层层嵌套。"""
        Barrage.PLANE = Plane()
        Barrage.BULLET = Bullet()
        plane_position = np.asarray(Barrage.PLANE.position, dtype=np.float32)
        plane_velocity = np.asarray(Barrage.PLANE.velocity, dtype=np.float32)
        Bullet.LIST = []
        Barrage.OPENING_SPAWNED_BATCHES = 0
        Barrage.BULLET._append_next_opening_batch(
            plane_position,
            plane_velocity,
        )
        Barrage.BULLET_FIELD_CONFIG = BulletFieldConfig(
            wall_collision=Barrage.COLLISION,
            screen_width=Barrage.SCREEN_WIDTH,
            screen_height=Barrage.SCREEN_HEIGHT,
            bullet_speed=Barrage.BULLET_SPEED,
            targeted_probability=Barrage.TARGETED_BULLET_PROBABILITY,
            prediction_scale_min=Barrage.TARGETED_PREDICTION_SCALE_MIN,
            prediction_scale_max=Barrage.TARGETED_PREDICTION_SCALE_MAX,
            angular_noise=Barrage.TARGETED_ANGULAR_NOISE,
        )
        Barrage.OPENING_EFFECTIVE_PHASE_SECONDS = (
            (Barrage.OPENING_BATCH_COUNT - 1)
            * Barrage.OPENING_BATCH_INTERVAL_SECONDS
        )

        Barrage.TimeStart = 0.0
        Barrage.TimeNow = 0.0
        Barrage.ALIVE_PHYSICS_STEPS = 0
        Barrage.TimeColor = "#ffffff"
        Barrage.TimeText = None
        Barrage.TimeTextKey = None
        Barrage.AI_ACTION = 0
        if Barrage.AI_CONTROLLER is not None:
            if Barrage.AI_PIPELINE is not None:
                Barrage.AI_PIPELINE.reset()
            else:
                Barrage.AI_CONTROLLER.reset()
            if Barrage.window is not None and Barrage.AI_PIPELINE is None:
                Barrage.render_world()
                Barrage.AI_ACTION = Barrage.AI_CONTROLLER.prime_surface(
                    Barrage.window
                )

    @staticmethod
    def render_world():
        """Render only game state, keeping UI glyphs out of AI captures."""
        render_world_surface(
            Barrage.window,
            Barrage.PLANE.image,
            np.asarray(Barrage.PLANE.position, dtype=np.float32),
            Barrage.BULLET.image,
            np.asarray([bullet[:2] for bullet in Bullet.LIST], dtype=np.float32),
        )

    @staticmethod
    def get_time():
        if not Barrage.KEY:
            Barrage.TimeColor = "#CD7F32"
            Barrage.window.blit(Barrage.RestartText, (5, Barrage.TimeSize))

        # 固定120 Hz物理更新，每12个存活物理步计1分。
        text = str(Barrage.ALIVE_PHYSICS_STEPS * 10 // Barrage.PHYSICS_FPS)
        text_key = (text, Barrage.TimeColor)
        # 数字或颜色不变时复用已经渲染的画面
        if text_key != Barrage.TimeTextKey:
            Barrage.TimeText = Barrage.TimeFont.render(text, True, Barrage.TimeColor)
            Barrage.TimeTextKey = text_key
        Barrage.window.blit(Barrage.TimeText, (5, 0))

    @classmethod
    def get_event(cls):
        key_list = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                if Barrage.AI_PIPELINE is not None:
                    Barrage.AI_PIPELINE.close()
                    Barrage.AI_PIPELINE = None
                pygame.quit()
                raise SystemExit
            if event.type == pygame.KEYDOWN:
                Barrage.PRESSED_SCANCODES.add(event.scancode)
            elif event.type == pygame.KEYUP:
                Barrage.PRESSED_SCANCODES.discard(event.scancode)
            elif event.type == pygame.WINDOWFOCUSLOST:
                Barrage.PRESSED_SCANCODES.clear()

        restarted = False
        switch_skin = False
        if not Barrage.KEY and (
            key_list[pygame.K_SPACE] or key_list[pygame.K_RETURN]
        ):
            Barrage.KEY = True
            Barrage.reset_game()
            restarted = True

        if Barrage.AI_CONTROLLER is not None:
            direction = ACTION_VECTORS[Barrage.AI_ACTION].copy()
        else:
            physical = Barrage.PRESSED_SCANCODES
            direction = normalized_direction(
                float(
                    key_list[pygame.K_RIGHT]
                    or key_list[pygame.K_d]
                    or pygame.KSCAN_D in physical
                )
                - float(
                    key_list[pygame.K_LEFT]
                    or key_list[pygame.K_a]
                    or pygame.KSCAN_A in physical
                ),
                float(
                    key_list[pygame.K_DOWN]
                    or key_list[pygame.K_s]
                    or pygame.KSCAN_S in physical
                )
                - float(
                    key_list[pygame.K_UP]
                    or key_list[pygame.K_w]
                    or pygame.KSCAN_W in physical
                ),
            )
            if Barrage.KEY:
                for key in range(10):
                    if key_list[key + pygame.K_0] and key != Plane.SKIN:
                        Plane.SKIN = key
                        switch_skin = True
                        break

        return switch_skin, direction, restarted

    @staticmethod
    def advance_physics(direction, delta_time):
        """Advance movement and collision by exactly one fixed physics step."""
        if not Barrage.KEY:
            return
        Barrage.PLANE.move(direction, delta_time)
        Barrage.BULLET.update(Barrage.PLANE, delta_time)
        Barrage.ALIVE_PHYSICS_STEPS += 1
        Barrage.TimeNow += delta_time


class Plane:
    # 默认使用独立的 plane(0).gif 外观
    SKIN = 0
    IMAGE = None

    def __init__(self):
        if Plane.SKIN:
            Plane.IMAGE = pygame.image.load(
                str(PROJECT_ROOT / "image" / f"plane({Plane.SKIN}).gif")
            ).convert_alpha()
        else:
            Plane.IMAGE = pygame.image.load(
                str(PROJECT_ROOT / "image" / "plane(0).gif")
            ).convert_alpha()
        self.image = Plane.IMAGE
        self.rect = Plane.IMAGE.get_rect()
        self.rect.center = 0.5 * Barrage.SCREEN_WIDTH, 0.5 * Barrage.SCREEN_HEIGHT
        self.position = pygame.Vector2(self.rect.center)
        self.velocity = pygame.Vector2(0.0, 0.0)
        # 像素遮罩只在图片变化时生成一次
        self.mask = pygame.mask.from_surface(self.image)

    def change_skin(self):
        center = self.rect.center
        Plane.IMAGE = pygame.image.load(
            str(PROJECT_ROOT / "image" / f"plane({Plane.SKIN}).gif")
        ).convert_alpha()
        self.image = Plane.IMAGE
        self.rect = self.image.get_rect(center=center)
        self.rect.clamp_ip(pygame.Rect(0, 0, Barrage.SCREEN_WIDTH, Barrage.SCREEN_HEIGHT))
        self.position.update(self.rect.center)
        self.mask = pygame.mask.from_surface(self.image)

    def move(self, direction, delta_time):
        # 使用浮点位置和固定物理子步，使移动不再依赖实际渲染帧率。
        # Accept the old four-boolean test/human representation as well as the
        # shared normalized two-vector used by the AI.
        if len(direction) == 4:
            vector = normalized_direction(
                float(direction[1]) - float(direction[0]),
                float(direction[3]) - float(direction[2]),
            )
        else:
            vector = normalized_direction(float(direction[0]), float(direction[1]))
        position, velocity = advance_plane(
            np.asarray(self.position, dtype=np.float32),
            vector,
            Barrage.PLANE_SPEED,
            delta_time,
            np.asarray(self.rect.size, dtype=np.float32) / 2.0,
            Barrage.SCREEN_WIDTH,
            Barrage.SCREEN_HEIGHT,
        )
        self.position.update(float(position[0]), float(position[1]))
        self.velocity.update(float(velocity[0]), float(velocity[1]))
        self.rect.center = self.position


class Bullet:
    LIST = list()

    def __init__(self):
        self.image = pygame.image.load(
            str(PROJECT_ROOT / "image" / f"bullet({Barrage.BULLET_SIZE}).gif")
        ).convert_alpha()
        self.rect = self.image.get_rect()
        # 所有子弹共用同一张图片，因此遮罩也只需创建一次
        self.mask = pygame.mask.from_surface(self.image)

    def _append_next_opening_batch(
        self,
        plane_position: np.ndarray,
        plane_velocity: np.ndarray,
    ) -> None:
        batch_index = Barrage.OPENING_SPAWNED_BATCHES
        if batch_index >= Barrage.OPENING_BATCH_COUNT:
            return
        count = opening_batch_size(
            Barrage.QUANTITY,
            batch_index,
            Barrage.OPENING_BATCH_COUNT,
        )
        positions, velocities, targeted = spawn_bullets(
            count,
            Barrage.SCREEN_WIDTH,
            Barrage.SCREEN_HEIGHT,
            Barrage.BULLET_SPEED,
            plane_position,
            plane_velocity,
            Barrage.TARGETED_BULLET_PROBABILITY,
            Barrage.TARGETED_PREDICTION_SCALE_MIN,
            Barrage.TARGETED_PREDICTION_SCALE_MAX,
            Barrage.TARGETED_ANGULAR_NOISE,
            Barrage.RNG,
        )
        Bullet.LIST.extend(
            [
                [
                    float(position[0]),
                    float(position[1]),
                    float(velocity[0]),
                    float(velocity[1]),
                    bool(is_targeted),
                ]
                for position, velocity, is_targeted in zip(
                    positions, velocities, targeted
                )
            ]
        )
        Barrage.OPENING_SPAWNED_BATCHES += 1

    def _append_due_opening_batches(self, plane) -> None:
        elapsed_steps = Barrage.ALIVE_PHYSICS_STEPS + 1
        while (
            Barrage.OPENING_SPAWNED_BATCHES < Barrage.OPENING_BATCH_COUNT
            and elapsed_steps >= opening_batch_physics_step(
                Barrage.OPENING_SPAWNED_BATCHES,
                Barrage.PHYSICS_FPS,
                Barrage.OPENING_BATCH_INTERVAL_SECONDS,
            )
        ):
            self._append_next_opening_batch(
                np.asarray(plane.position, dtype=np.float32),
                np.asarray(plane.velocity, dtype=np.float32),
            )

    def update(self, plane, delta_time):
        if Barrage.KEY:
            if Bullet.LIST:
                bullet_state = np.asarray(Bullet.LIST, dtype=np.float32)
                positions = bullet_state[:, :2]
                velocities = bullet_state[:, 2:4]
                targeted = bullet_state[:, 4].astype(np.bool_)
                advance_bullet_field(
                    positions,
                    velocities,
                    targeted,
                    np.asarray(plane.position, dtype=np.float32),
                    np.asarray(plane.velocity, dtype=np.float32),
                    Barrage.RNG,
                    delta_time,
                    Barrage.BULLET_FIELD_CONFIG,
                )
                # Convert NumPy scalars in C while retaining the public list and
                # each row's identity for callers that keep or edit references.
                for bullet_data, values, is_targeted in zip(
                    Bullet.LIST, bullet_state[:, :4].tolist(), targeted.tolist()
                ):
                    bullet_data[:4] = values
                    bullet_data[4] = is_targeted

            self._append_due_opening_batches(plane)
            positions = np.asarray(
                [bullet[:2] for bullet in Bullet.LIST], dtype=np.float32
            ).reshape(-1, 2)

            if Barrage.INVINCIBLE and len(colliding_bullet_indices(
                np.asarray(plane.position, dtype=np.float32),
                positions,
                plane.image,
                self.image,
                plane.mask,
                self.mask,
            )):
                Barrage.KEY = False

class Music:
    def __init__(self, bg):
        self.started = False
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(str(bg))
            pygame.mixer.music.play(-1)
            self.started = True
        except (OSError, pygame.error) as error:
            # Some Windows systems have no active WASAPI output endpoint.
            # Music is optional, so keep the game playable without audio.
            Barrage.MUSIC = False
            print(f"音乐已自动关闭：{error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Barrage game")
    parser.add_argument(
        "-ai",
        "--ai",
        action="store_true",
        help="use the default best AI model in the window controller",
    )
    parser.add_argument(
        "--analytic-shield",
        action=argparse.BooleanOptionalAction,
        default=PRODUCTION_ANALYTIC_SHIELD,
        help="enable an optional independent image-geometry constraint",
    )
    parser.add_argument("--bullet-size", type=int, default=TARGET_TASK.bullet_size)
    parser.add_argument("--bullets", type=int, default=TARGET_TASK.bullet_count)
    parser.add_argument("--plane-speed", type=float, default=TARGET_TASK.bullet_speed)
    parser.add_argument("--bullet-speed", type=float, default=TARGET_TASK.bullet_speed)
    parser.add_argument(
        "--targeted-probability",
        type=float,
        default=TARGET_TASK.targeted_bullet_probability,
    )
    parser.add_argument("--targeted-prediction-min", type=float, default=0.65)
    parser.add_argument("--targeted-prediction-max", type=float, default=1.0)
    parser.add_argument("--targeted-angular-noise", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--skin", type=int, default=0)
    parser.add_argument("--pixel-guard", choices=("off", "refined", "receding"), default="receding")
    parser.add_argument("--no-music", action="store_true")
    parser.add_argument("--latency-test-seconds", type=float, default=0.0)
    parser.add_argument("--latency-allow-damage", action="store_true",
                        help="retain real collisions during a timed visible-window test")
    parser.add_argument("--latency-output", default="")
    args = parser.parse_args()
    if args.ai:
        if args.pixel_guard != "off" and (args.skin != 0 or args.bullet_size != 5 or args.plane_speed != 240 or args.bullet_speed != 240):
            parser.error("pixel guard requires skin 0, bullet size 5, and plane/bullet speeds 240; use --pixel-guard off for other settings")
        import os
        import torch
        from barrage_rl.live_screen import LiveVisualController

        # Single-frame latency/parity sweep on the reference 10-core host.
        # Bound the pool on smaller machines; training keeps its own settings.
        torch.set_num_threads(min(10, os.cpu_count() or 1))
        Barrage.SCREEN_WIDTH = FIXED_SCREEN_WIDTH
        Barrage.SCREEN_HEIGHT = FIXED_SCREEN_HEIGHT
        Barrage.BULLET_SIZE = args.bullet_size
        Barrage.QUANTITY = args.bullets
        Barrage.PLANE_SPEED = args.plane_speed
        Barrage.BULLET_SPEED = args.bullet_speed
        Barrage.TARGETED_BULLET_PROBABILITY = args.targeted_probability
        Barrage.TARGETED_PREDICTION_SCALE_MIN = args.targeted_prediction_min
        Barrage.TARGETED_PREDICTION_SCALE_MAX = args.targeted_prediction_max
        Barrage.TARGETED_ANGULAR_NOISE = args.targeted_angular_noise
        Barrage.RNG = np.random.default_rng(args.seed)
        Barrage.MUSIC = not args.no_music
        Barrage.LATENCY_TEST_SECONDS = max(0.0, args.latency_test_seconds)
        Barrage.LATENCY_OUTPUT = args.latency_output or None
        if Barrage.LATENCY_TEST_SECONDS > 0.0 and not args.latency_allow_damage:
            # Keep the visible diagnostic alive for its full requested duration.
            Barrage.INVINCIBLE = False
        Barrage.TimeSize = int(
            math.sqrt(FIXED_SCREEN_WIDTH * FIXED_SCREEN_HEIGHT) / 25
        )
        Plane.SKIN = args.skin
        Barrage.AI_CONTROLLER = LiveVisualController(
            str(DEFAULT_AI_CHECKPOINT),
            device_name="cpu",
            analytic_shield=args.analytic_shield,

        )
        if args.pixel_guard == "receding":
            from barrage_rl.deployment import configure_image_controller
            configure_image_controller(Barrage.AI_CONTROLLER.agent, "receding", search_workers=9)
        elif args.pixel_guard == "refined":
            from tools.pixel_guard_refined import install_refined_guard
            install_refined_guard(Barrage.AI_CONTROLLER.agent)
        Barrage.start_game()
    else:
        if settings():
            Barrage.start_game()
