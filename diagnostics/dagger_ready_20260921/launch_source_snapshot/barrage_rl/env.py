"""只向智能体提供像素帧的 Gymnasium 弹幕环境。"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces

from .task_spec import TARGET_TASK

from .timing import DECISION_HZ, DEFAULT_ACTION_REPEAT, PHYSICS_FPS

from .runtime_core import (
    ACTION_VECTORS,
    BulletFieldConfig,
    OPENING_BATCH_COUNT,
    OPENING_BATCH_INTERVAL_SECONDS,
    advance_bullet_field,
    advance_plane,
    colliding_bullet_indices as find_colliding_bullet_indices,
    opening_batch_physics_step,
    opening_batch_size,
    render_world_surface,
    respawn_bullet_indices,
    sample_episode_parameters,
    spawn_bullets,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class EnvSnapshot:
    """Complete restorable simulator state used by reset validation and MPC."""

    plane_position: np.ndarray
    plane_velocity: np.ndarray
    bullet_positions: np.ndarray
    bullet_velocities: np.ndarray
    bullet_is_targeted: np.ndarray
    frames: Tuple[np.ndarray, ...]
    episode_steps: int
    physics_steps: int
    opening_spawned_batches: int
    rng_state: Dict[str, Any]


@dataclass
class SimulationSnapshot:
    """Lightweight exact state for branching without pixel-frame copies."""

    plane_position: np.ndarray
    plane_velocity: np.ndarray
    bullet_positions: np.ndarray
    bullet_velocities: np.ndarray
    bullet_is_targeted: np.ndarray
    episode_steps: int
    physics_steps: int
    opening_spawned_batches: int
    rng_state: Dict[str, Any]


class BarrageVisionEnv(gym.Env):
    """Barrage 的无窗口训练环境。

    游戏内部坐标只用于物理计算、画面渲染和奖励判定；智能体能够获得的
    observation 始终只有连续的灰度像素帧，不会收到飞机或子弹坐标。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": DECISION_HZ}
    BACKGROUND_INTENSITY = 0
    PLANE_INTENSITY = 96
    BULLET_INTENSITY = 255
    BULLET_SIZE_INTENSITY_BASE = 160
    MIN_PLANE_OBSERVATION_SIZE = 4
    BULLET_OBSERVATION_SIZE = 3

    # 0不动，其余动作依次为四方向和四个斜方向
    ACTIONS = ACTION_VECTORS

    _plane_surface = None
    _plane_mask = None
    _bullet_assets: Dict[int, Tuple[pygame.Surface, pygame.mask.Mask]] = {}

    def __init__(
        self,
        bullet_count: int = TARGET_TASK.bullet_count,
        bullet_size: int = 5,
        bullet_size_min: Optional[int] = None,
        bullet_size_max: Optional[int] = None,
        bullet_speed_min: float = 240.0,
        bullet_speed_max: float = 240.0,
        observation_size: int = 96,
        frame_stack: int = 4,
        action_repeat: int = DEFAULT_ACTION_REPEAT,
        max_episode_seconds: float = 60.0,
        wall_collision: bool = False,
        render_mode: Optional[str] = None,
        screen_width: int = 820,
        screen_height: int = 820,
        randomize_initial_phase: bool = True,
        initial_phase_min_seconds: float = 0.75,
        initial_phase_max_seconds: float = 3.0,
        dense_reward_scale: float = 0.25,
        danger_horizon_seconds: float = 2.0,
        targeted_bullet_probability: float = 0.10,
        targeted_prediction_scale_min: float = 0.65,
        targeted_prediction_scale_max: float = 1.0,
        targeted_angular_noise: float = 0.08,
        recoverability_seconds: float = 0.60,
    ) -> None:
        super().__init__()
        if bullet_count < 1:
            raise ValueError("bullet_count 必须大于0")
        bullet_size_min = bullet_size if bullet_size_min is None else bullet_size_min
        bullet_size_max = bullet_size if bullet_size_max is None else bullet_size_max
        if bullet_size_min not in range(1, 11) or bullet_size_max not in range(1, 11):
            raise ValueError("子弹大小范围必须在1到10之间")
        if bullet_size_max < bullet_size_min:
            raise ValueError("bullet_size_max 不能小于 bullet_size_min")
        if bullet_speed_min <= 0 or bullet_speed_max < bullet_speed_min:
            raise ValueError("子弹速度范围无效")
        if observation_size < 48:
            raise ValueError("observation_size 过小，细小子弹可能无法显示")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode 必须是 None、human 或 rgb_array")
        if initial_phase_min_seconds < 0:
            raise ValueError("initial_phase_min_seconds 不能小于0")
        if initial_phase_max_seconds < initial_phase_min_seconds:
            raise ValueError("initial_phase_max_seconds 不能小于最小值")
        if dense_reward_scale < 0:
            raise ValueError("dense_reward_scale 不能小于0")
        if danger_horizon_seconds <= 0:
            raise ValueError("danger_horizon_seconds 必须大于0")
        if not 0.0 <= targeted_bullet_probability <= 1.0:
            raise ValueError("targeted_bullet_probability 必须在0到1之间")
        if targeted_prediction_scale_min < 0:
            raise ValueError("targeted_prediction_scale_min 不能小于0")
        if targeted_prediction_scale_max < targeted_prediction_scale_min:
            raise ValueError("targeted_prediction_scale_max 不能小于最小值")
        if targeted_angular_noise < 0:
            raise ValueError("targeted_angular_noise 不能小于0")
        if recoverability_seconds <= 0:
            raise ValueError("recoverability_seconds must be positive")
        if action_repeat < 1 or max_episode_seconds <= 0:
            raise ValueError("action_repeat and max_episode_seconds must be positive")

        self.screen_width = int(screen_width)
        self.screen_height = int(screen_height)
        self.bullet_count = int(bullet_count)
        self.bullet_size = int(bullet_size)
        self.bullet_size_min = int(bullet_size_min)
        self.bullet_size_max = int(bullet_size_max)
        self.bullet_speed_min = float(bullet_speed_min)
        self.bullet_speed_max = float(bullet_speed_max)
        self.bullet_speed = float(bullet_speed_min)
        self.observation_size = int(observation_size)
        self.frame_stack = int(frame_stack)
        self.action_repeat = int(action_repeat)
        self.max_episode_seconds = float(max_episode_seconds)
        self.max_episode_physics_steps = int(np.ceil(max_episode_seconds * PHYSICS_FPS))
        self.wall_collision = bool(wall_collision)
        self.render_mode = render_mode
        self.randomize_initial_phase = bool(randomize_initial_phase)
        self.initial_phase_min_seconds = float(initial_phase_min_seconds)
        self.initial_phase_max_seconds = float(initial_phase_max_seconds)
        self.dense_reward_scale = float(dense_reward_scale)
        self.danger_horizon_seconds = float(danger_horizon_seconds)
        self.targeted_bullet_probability = float(targeted_bullet_probability)
        self.targeted_prediction_scale_min = float(targeted_prediction_scale_min)
        self.targeted_prediction_scale_max = float(targeted_prediction_scale_max)
        self.targeted_angular_noise = float(targeted_angular_noise)
        self.recoverability_seconds = float(recoverability_seconds)
        self.reset_fallback_count = 0
        self.opening_effective_phase_seconds = 0.0
        self.opening_spawned_batches = OPENING_BATCH_COUNT

        self.physics_fps = PHYSICS_FPS
        self.delta_time = 1.0 / self.physics_fps
        self.speed = 240.0
        self.plane_position = np.zeros(2, dtype=np.float32)
        self.plane_velocity = np.zeros(2, dtype=np.float32)
        self.bullet_positions = np.empty((self.bullet_count, 2), dtype=np.float32)
        self.bullet_velocities = np.empty((self.bullet_count, 2), dtype=np.float32)
        self.bullet_is_targeted = np.zeros(self.bullet_count, dtype=np.bool_)
        self.frames: Deque[np.ndarray] = deque(maxlen=self.frame_stack)
        self.episode_steps = 0
        self.physics_steps = 0
        self._pending_bullet_count = self.bullet_count

        self.action_space = spaces.Discrete(len(self.ACTIONS))
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.frame_stack, self.observation_size, self.observation_size),
            dtype=np.uint8,
        )

        self._window: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None
        self._load_collision_assets()

    def _load_collision_assets(self) -> None:
        """碰撞遮罩只加载一次，训练过程中不做重复图片解码。"""
        if BarrageVisionEnv._plane_surface is None:
            plane = pygame.image.load(str(PROJECT_ROOT / "image" / "plane(0).gif"))
            BarrageVisionEnv._plane_surface = plane
            BarrageVisionEnv._plane_mask = pygame.mask.from_surface(plane)

        if self.bullet_size not in BarrageVisionEnv._bullet_assets:
            bullet = pygame.image.load(
                str(PROJECT_ROOT / "image" / ("bullet(%d).gif" % self.bullet_size))
            )
            BarrageVisionEnv._bullet_assets[self.bullet_size] = (
                bullet,
                pygame.mask.from_surface(bullet),
            )

        self.plane_surface = BarrageVisionEnv._plane_surface
        self.plane_mask = BarrageVisionEnv._plane_mask
        self.bullet_surface, self.bullet_mask = BarrageVisionEnv._bullet_assets[
            self.bullet_size
        ]
        self.plane_size = np.asarray(self.plane_surface.get_size(), dtype=np.float32)

    def set_bullet_count(self, bullet_count: int) -> None:
        """在下一局开始时应用课程学习的新子弹数量。"""
        self._pending_bullet_count = max(1, int(bullet_count))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if options and "bullet_count" in options:
            self.set_bullet_count(int(options["bullet_count"]))
        self.bullet_count = self._pending_bullet_count
        self.bullet_size, self.bullet_speed = sample_episode_parameters(
            self.bullet_size_min,
            self.bullet_size_max,
            self.bullet_speed_min,
            self.bullet_speed_max,
            self.np_random,
        )
        self.bullet_field_config = BulletFieldConfig(
            wall_collision=self.wall_collision,
            screen_width=self.screen_width,
            screen_height=self.screen_height,
            bullet_speed=self.bullet_speed,
            targeted_probability=self.targeted_bullet_probability,
            prediction_scale_min=self.targeted_prediction_scale_min,
            prediction_scale_max=self.targeted_prediction_scale_max,
            angular_noise=self.targeted_angular_noise,
        )
        reset_mode = "mid_episode" if self.randomize_initial_phase else "deployment"
        if options and "reset_mode" in options:
            reset_mode = str(options["reset_mode"])
        self._load_collision_assets()

        self.plane_position[:] = (self.screen_width / 2, self.screen_height / 2)
        self.plane_velocity.fill(0.0)
        self.bullet_positions = np.empty((self.bullet_count, 2), dtype=np.float32)
        self.bullet_velocities = np.empty((self.bullet_count, 2), dtype=np.float32)
        self.bullet_is_targeted = np.zeros(self.bullet_count, dtype=np.bool_)

        self.episode_steps = 0
        self.physics_steps = 0
        self.opening_effective_phase_seconds = 0.0
        self.opening_spawned_batches = OPENING_BATCH_COUNT
        if reset_mode == "deployment":
            self._initialize_deployment_phase()
        elif reset_mode in ("mid_episode", "recoverable_hard"):
            if not self._initialize_recoverable_phase(reset_mode):
                self.reset_fallback_count += 1
                self._initialize_deployment_phase()
                reset_mode = "deployment_fallback"
        else:
            raise ValueError(f"unknown reset_mode: {reset_mode}")
        self.reset_mode = reset_mode

        return self._get_observation(), self._get_info()

    def _initialize_deployment_phase(self) -> None:
        """Match deployment with ten ungated batches from t=0.0 through t=0.9."""
        self.bullet_positions = np.empty((0, 2), dtype=np.float32)
        self.bullet_velocities = np.empty((0, 2), dtype=np.float32)
        self.bullet_is_targeted = np.empty(0, dtype=np.bool_)
        self.opening_spawned_batches = 0
        self._append_next_opening_batch()
        self.opening_effective_phase_seconds = (
            (OPENING_BATCH_COUNT - 1) * OPENING_BATCH_INTERVAL_SECONDS
        )
        first_frame = self._make_frame()
        self.frames.clear()
        for _ in range(self.frame_stack):
            self.frames.append(first_frame.copy())

    def _append_next_opening_batch(self) -> None:
        batch_index = self.opening_spawned_batches
        if batch_index >= OPENING_BATCH_COUNT:
            return
        count = opening_batch_size(
            self.bullet_count,
            batch_index,
            OPENING_BATCH_COUNT,
        )
        positions, velocities, targeted = spawn_bullets(
            count,
            self.screen_width,
            self.screen_height,
            self.bullet_speed,
            self.plane_position,
            self.plane_velocity,
            self.targeted_bullet_probability,
            self.targeted_prediction_scale_min,
            self.targeted_prediction_scale_max,
            self.targeted_angular_noise,
            self.np_random,
        )
        self.bullet_positions = np.concatenate((self.bullet_positions, positions))
        self.bullet_velocities = np.concatenate((self.bullet_velocities, velocities))
        self.bullet_is_targeted = np.concatenate((
            self.bullet_is_targeted,
            targeted,
        ))
        self.opening_spawned_batches += 1

    def _append_due_opening_batches(self, elapsed_steps: int) -> None:
        while (
            self.opening_spawned_batches < OPENING_BATCH_COUNT
            and elapsed_steps >= opening_batch_physics_step(
                self.opening_spawned_batches,
                self.physics_fps,
                OPENING_BATCH_INTERVAL_SECONDS,
            )
        ):
            self._append_next_opening_batch()

    def _initialize_recoverable_phase(self, mode: str) -> bool:
        """Build moving history while rejecting dead-on-arrival layouts."""
        minimum_decisions = max(
            self.frame_stack - 1,
            int(round(self.initial_phase_min_seconds / self.decision_dt)),
        )
        maximum_decisions = max(
            minimum_decisions,
            int(round(self.initial_phase_max_seconds / self.decision_dt)),
        )
        best_snapshot: Optional[EnvSnapshot] = None
        best_safe_count = len(self.ACTIONS) + 1
        for _ in range(128):
            self.plane_position[:] = (self.screen_width / 2, self.screen_height / 2)
            self.plane_velocity.fill(0.0)
            self._spawn_bullets(np.arange(self.bullet_count))
            self.frames.clear()
            self.frames.append(self._make_frame())
            warmup_decisions = int(
                self.np_random.integers(minimum_decisions, maximum_decisions + 1)
            )
            valid = True
            for _decision in range(warmup_decisions):
                chosen = None
                for action in self.np_random.permutation(len(self.ACTIONS)):
                    if self._action_survives(int(action), 1):
                        chosen = int(action)
                        break
                if chosen is None:
                    valid = False
                    break
                for _ in range(self.action_repeat):
                    self._move_plane(self.ACTIONS[chosen])
                    self._move_bullets()
                    if self._has_collision():
                        valid = False
                        break
                if not valid:
                    break
                self.frames.append(self._make_frame())
            if not valid:
                continue
            while len(self.frames) < self.frame_stack:
                self.frames.appendleft(self.frames[0].copy())
            safe_count = self._safe_action_count(self.recoverability_seconds)
            if safe_count == 0:
                continue
            if safe_count < best_safe_count:
                best_snapshot = self.capture_state()
                best_safe_count = safe_count
            if mode == "mid_episode" and safe_count >= 2:
                return True
            if mode == "recoverable_hard" and safe_count <= 3:
                return True
        if best_snapshot is not None:
            self.restore_state(best_snapshot)
            return mode == "mid_episode" or best_safe_count <= 4
        return False

    @property
    def max_episode_steps(self) -> int:
        return int(np.ceil(self.max_episode_physics_steps / self.action_repeat))

    @property
    def decision_dt(self) -> float:
        return self.action_repeat * self.delta_time

    def capture_state(self) -> EnvSnapshot:
        return EnvSnapshot(
            plane_position=self.plane_position.copy(),
            plane_velocity=self.plane_velocity.copy(),
            bullet_positions=self.bullet_positions.copy(),
            bullet_velocities=self.bullet_velocities.copy(),
            bullet_is_targeted=self.bullet_is_targeted.copy(),
            frames=tuple(frame.copy() for frame in self.frames),
            episode_steps=int(self.episode_steps),
            physics_steps=int(self.physics_steps),
            opening_spawned_batches=int(self.opening_spawned_batches),
            rng_state=deepcopy(self.np_random.bit_generator.state),
        )

    def restore_state(self, snapshot: EnvSnapshot) -> None:
        self.plane_position[:] = snapshot.plane_position
        self.plane_velocity[:] = snapshot.plane_velocity
        self.bullet_positions = snapshot.bullet_positions.copy()
        self.bullet_velocities = snapshot.bullet_velocities.copy()
        self.bullet_is_targeted = snapshot.bullet_is_targeted.copy()
        self.frames.clear()
        self.frames.extend(frame.copy() for frame in snapshot.frames)
        self.episode_steps = int(snapshot.episode_steps)
        self.physics_steps = int(snapshot.physics_steps)
        self.opening_spawned_batches = int(snapshot.opening_spawned_batches)
        self.np_random.bit_generator.state = deepcopy(snapshot.rng_state)

    def capture_simulation_state(self) -> SimulationSnapshot:
        """Capture exact physics/RNG state while omitting immutable MPC frames."""
        return SimulationSnapshot(
            plane_position=self.plane_position.copy(),
            plane_velocity=self.plane_velocity.copy(),
            bullet_positions=self.bullet_positions.copy(),
            bullet_velocities=self.bullet_velocities.copy(),
            bullet_is_targeted=self.bullet_is_targeted.copy(),
            episode_steps=int(self.episode_steps),
            physics_steps=int(self.physics_steps),
            opening_spawned_batches=int(self.opening_spawned_batches),
            rng_state=deepcopy(self.np_random.bit_generator.state),
        )

    def restore_simulation_state(self, snapshot: SimulationSnapshot) -> None:
        """Restore a lightweight branch without touching observation frames."""
        self.plane_position[:] = snapshot.plane_position
        self.plane_velocity[:] = snapshot.plane_velocity
        self.bullet_positions = snapshot.bullet_positions.copy()
        self.bullet_velocities = snapshot.bullet_velocities.copy()
        self.bullet_is_targeted = snapshot.bullet_is_targeted.copy()
        self.episode_steps = int(snapshot.episode_steps)
        self.physics_steps = int(snapshot.physics_steps)
        self.opening_spawned_batches = int(snapshot.opening_spawned_batches)
        self.np_random.bit_generator.state = deepcopy(snapshot.rng_state)

    def _action_survives(self, action: int, decisions: int) -> bool:
        snapshot = self.capture_state()
        try:
            for _ in range(max(1, int(decisions))):
                if self.simulate_action(int(action)):
                    return False
            return True
        finally:
            self.restore_state(snapshot)

    def _safe_action_count(self, horizon_seconds: float) -> int:
        decisions = max(1, int(np.ceil(float(horizon_seconds) / self.decision_dt)))
        return sum(
            self._action_survives(action, decisions)
            for action in range(len(self.ACTIONS))
        )

    def _spawn_bullets(self, indices: np.ndarray) -> None:
        respawn_bullet_indices(
            self.bullet_positions,
            self.bullet_velocities,
            self.bullet_is_targeted,
            indices,
            self.plane_position,
            self.plane_velocity,
            self.np_random,
            self.bullet_field_config,
        )

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError("无效动作: %s" % action)

        direction = self.ACTIONS[int(action)]
        danger_before = self._danger_potential()
        terminated = False
        survived_substeps = 0
        for _ in range(self.action_repeat):
            self._move_plane(direction)
            self._move_bullets()
            self.physics_steps += 1
            if self._has_collision():
                terminated = True
                break
            survived_substeps += 1
            if self.physics_steps >= self.max_episode_physics_steps:
                break

        self.episode_steps += 1
        truncated = (not terminated) and self.physics_steps >= self.max_episode_physics_steps
        self.frames.append(self._make_frame())

        # 奖励保持简单，减少智能体贴边或钻奖励漏洞的机会
        reward = 0.01 * survived_substeps / self.action_repeat
        if terminated:
            reward = -1.0
        else:
            danger_after = self._danger_potential()
            reward += self.dense_reward_scale * (danger_before - danger_after)

        if self.render_mode == "human":
            self.render()

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def simulate_action(
        self,
        action: int,
        *,
        include_scheduled_opening: bool = True,
        include_respawns: bool = True,
    ) -> bool:
        """Advance one exact decision without rendering/reward bookkeeping.

        The MPC teacher surrounds calls with lightweight simulation snapshots.
        Avoiding semantic-frame construction makes exact branching practical
        while retaining the same physics, respawn, target aim, and mask collision.
        Image-causal teacher branches can exclude every bullet that has not
        appeared in the current observation yet: scheduled opening batches and
        random replacements for bullets that leave the frame. Real ``step``
        calls keep both lifecycle mechanisms enabled.
        """
        direction = self.ACTIONS[int(action)]
        for _ in range(self.action_repeat):
            self._move_plane(direction)
            self._move_bullets(
                include_scheduled_opening=include_scheduled_opening,
                include_respawns=include_respawns,
            )
            self.physics_steps += 1
            if self._has_collision():
                return True
        return False

    def _danger_potential(self) -> float:
        """根据未来最近交会点估计危险度，仅用于奖励，不进入观测。"""
        relative_position = self.bullet_positions - self.plane_position
        velocity_squared = np.sum(self.bullet_velocities ** 2, axis=1)
        dot_product = np.sum(relative_position * self.bullet_velocities, axis=1)
        approaching = dot_product < 0.0
        time_to_closest = np.clip(
            -dot_product / np.maximum(velocity_squared, 1e-6),
            0.0,
            self.danger_horizon_seconds,
        )
        closest_offset = (
            relative_position
            + self.bullet_velocities * time_to_closest[:, None]
        )
        closest_distance = np.linalg.norm(closest_offset, axis=1)
        collision_radius = 0.5 * (
            max(float(self.plane_size[0]), float(self.plane_size[1]))
            + max(self.bullet_surface.get_size())
        )
        clearance = np.maximum(closest_distance - collision_radius, 0.0)
        danger = (
            approaching.astype(np.float32)
            * np.exp(-clearance / 36.0)
            * np.exp(-time_to_closest / 0.75)
        )
        if len(danger) == 0:
            return 0.0
        top_count = min(3, len(danger))
        return float(np.mean(np.partition(danger, -top_count)[-top_count:]))

    def _move_plane(self, direction: np.ndarray) -> None:
        position, velocity = advance_plane(
            self.plane_position,
            direction,
            self.speed,
            self.delta_time,
            self.plane_size / 2.0,
            self.screen_width,
            self.screen_height,
        )
        self.plane_position[:] = position
        self.plane_velocity[:] = velocity

    def _move_bullets(
        self,
        *,
        include_scheduled_opening: bool = True,
        include_respawns: bool = True,
    ) -> None:
        advance_bullet_field(
            self.bullet_positions,
            self.bullet_velocities,
            self.bullet_is_targeted,
            self.plane_position,
            self.plane_velocity,
            self.np_random,
            self.delta_time,
            self.bullet_field_config,
            respawn_outside=include_respawns,
        )
        if include_scheduled_opening:
            self._append_due_opening_batches(self.physics_steps + 1)

    def _has_collision(self) -> bool:
        return bool(len(find_colliding_bullet_indices(
            self.plane_position,
            self.bullet_positions,
            self.plane_surface,
            self.bullet_surface,
            self.plane_mask,
            self.bullet_mask,
        )))

    def colliding_bullet_indices(self) -> np.ndarray:
        """Return exact colliding bullets for privileged evaluation diagnostics.

        This method deliberately exposes simulator state only to offline
        attribution code.  The policy observation remains image-only.
        """
        return find_colliding_bullet_indices(
            self.plane_position,
            self.bullet_positions,
            self.plane_surface,
            self.bullet_surface,
            self.plane_mask,
            self.bullet_mask,
        )

    def _make_frame(self) -> np.ndarray:
        """直接生成网络分辨率画面，避免高分辨率截图和缩放延迟。"""
        size = self.observation_size
        frame = np.full(
            (size, size), self.BACKGROUND_INTENSITY, dtype=np.uint8
        )

        bullet_x = np.rint(
            self.bullet_positions[:, 0] * (size - 1) / self.screen_width
        ).astype(np.int32)
        bullet_y = np.rint(
            self.bullet_positions[:, 1] * (size - 1) / self.screen_height
        ).astype(np.int32)
        bullet_x = np.clip(bullet_x, 0, size - 1)
        bullet_y = np.clip(bullet_y, 0, size - 1)
        # The halo carries the visible sprite size while a one-pixel white core
        # gives the set extractor an unambiguous center.  The previous renderer
        # drew every bullet as the same 3x3 square, making different collision
        # radii observationally indistinguishable to the student.
        bullet_radius = max(
            self.BULLET_OBSERVATION_SIZE // 2,
            int(np.ceil(self.bullet_size / 2.0)),
        )
        size_intensity = self.BULLET_SIZE_INTENSITY_BASE + self.bullet_size
        for x, y in zip(bullet_x, bullet_y):
            left = max(0, int(x) - bullet_radius)
            right = min(size, int(x) + bullet_radius + 1)
            top = max(0, int(y) - bullet_radius)
            bottom = min(size, int(y) + bullet_radius + 1)
            frame[top:bottom, left:right] = np.maximum(
                frame[top:bottom, left:right], size_intensity
            )

        # Draw centers after all halos so overlapping bullets cannot erase one
        # another's detection marker.
        frame[bullet_y, bullet_x] = self.BULLET_INTENSITY

        plane_x = int(round(self.plane_position[0] * (size - 1) / self.screen_width))
        plane_y = int(round(self.plane_position[1] * (size - 1) / self.screen_height))
        plane_width = max(
            self.MIN_PLANE_OBSERVATION_SIZE,
            int(round(self.plane_size[0] * size / self.screen_width)),
        )
        plane_height = max(
            self.MIN_PLANE_OBSERVATION_SIZE,
            int(round(self.plane_size[1] * size / self.screen_height)),
        )
        left = max(0, plane_x - plane_width // 2)
        right = min(size, left + plane_width)
        top = max(0, plane_y - plane_height // 2)
        bottom = min(size, top + plane_height)
        frame[top:bottom, left:right] = self.PLANE_INTENSITY
        return frame

    def _get_observation(self) -> np.ndarray:
        return np.stack(tuple(self.frames), axis=0)

    def _get_info(self) -> Dict[str, Any]:
        survival_seconds = self.physics_steps / self.physics_fps
        return {
            "survival_seconds": survival_seconds,
            # 与人工游戏一致：每存活0.1秒计1分
            "score": int(10 * survival_seconds),
            "bullet_count": self.bullet_count,
            "bullet_size": self.bullet_size,
            "bullet_speed": self.bullet_speed,
            "targeted_bullet_probability": self.targeted_bullet_probability,
            "reset_mode": getattr(self, "reset_mode", "unknown"),
            "reset_fallback_count": self.reset_fallback_count,
            "opening_effective_phase_seconds": self.opening_effective_phase_seconds,
            "opening_spawned_batches": self.opening_spawned_batches,
            "active_bullet_count": len(self.bullet_positions),
        }

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            frame = self.frames[-1]
            return np.repeat(frame[:, :, None], 3, axis=2)
        if self.render_mode != "human":
            return None

        if self._window is None:
            pygame.init()
            self._window = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )
            pygame.display.set_caption("Barrage CNN Agent")
            self._clock = pygame.time.Clock()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        render_world_surface(
            self._window,
            self.plane_surface,
            self.plane_position,
            self.bullet_surface,
            self.bullet_positions,
        )
        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(1.0 / self.decision_dt)
        return None

    def close(self) -> None:
        if self._window is not None:
            pygame.display.quit()
            self._window = None
            self._clock = None
