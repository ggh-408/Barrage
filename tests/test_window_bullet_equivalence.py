"""Exact array physics and mutable diagnostic rows of the window bullet adapter."""
from barrage_rl.timing import PHYSICS_DT, PHYSICS_FPS

from copy import deepcopy
import struct
from types import SimpleNamespace

import numpy as np
import pytest

import Barrage as game_module
from Barrage import Barrage, Bullet
from barrage_rl.runtime_core import (
    BulletFieldConfig, advance_bullet_field, initialize_opening_bullets,
)


@pytest.fixture(autouse=True)
def complete_opening_batches(monkeypatch):
    # These tests supply the complete field directly, without reset_game.
    monkeypatch.setattr(Barrage, 'OPENING_SPAWNED_BATCHES', Barrage.OPENING_BATCH_COUNT)


def _reference_writeback(rows, state, targeted):
    for row, position, velocity, flag in zip(rows, state[:, :2], state[:, 2:4], targeted):
        row[0] = float(position[0])
        row[1] = float(position[1])
        row[2] = float(velocity[0])
        row[3] = float(velocity[1])
        row[4] = bool(flag)


def _check_rows_exact(actual, expected):
    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert len(actual_row) == len(expected_row)
        for actual_value, expected_value in zip(actual_row[:4], expected_row[:4]):
            assert type(actual_value) is float
            assert struct.pack("!d", actual_value) == struct.pack("!d", expected_value)
        assert type(actual_row[4]) is bool
        assert actual_row[4] == expected_row[4]
        assert actual_row[5:] == expected_row[5:]


def test_array_update_keeps_exact_special_values_and_row_identity(monkeypatch):
    values = np.asarray([
        0.0, -0.0, np.inf, -np.inf, np.nan,
        np.nextafter(np.float32(0.0), np.float32(1.0)),
        np.finfo(np.float32).max, np.finfo(np.float32).tiny,
    ], dtype=np.float32)
    state = np.column_stack([np.roll(values, shift) for shift in range(4)])
    targeted = np.arange(len(values)) % 2 == 0
    from barrage_rl.window_bullets import WindowBulletField
    rows = WindowBulletField()
    rows.append_batch(np.zeros((len(values), 2), np.float32),
                      np.zeros((len(values), 2), np.float32), np.zeros(len(values), bool))
    expected = [list(row) for row in rows]
    original_rows = list(rows)
    _reference_writeback(expected, state, targeted)

    def replace_field(positions, velocities, flags, *args):
        positions[:] = state[:, :2]
        velocities[:] = state[:, 2:4]
        flags[:] = targeted

    monkeypatch.setattr(game_module, "advance_bullet_field", replace_field)
    monkeypatch.setattr(Bullet, "LIST", rows)
    monkeypatch.setattr(Barrage, "KEY", True)
    monkeypatch.setattr(Barrage, "INVINCIBLE", False)
    plane = SimpleNamespace(position=[410.0, 410.0], velocity=[0.0, 0.0])
    Bullet.update(object.__new__(Bullet), plane, PHYSICS_DT)
    assert Bullet.LIST is rows
    assert all(row is original for row, original in zip(rows, original_rows))
    _check_rows_exact(rows, expected)


@pytest.mark.parametrize("wall_collision", [False, True])
def test_window_field_matches_old_adapter_and_rng_over_respawns(monkeypatch, wall_collision):
    plane = SimpleNamespace(
        position=np.asarray([410.25, 409.75], dtype=np.float32),
        velocity=np.asarray([240.0, 0.0], dtype=np.float32),
    )
    rng = np.random.default_rng(1807)
    config = BulletFieldConfig(
        wall_collision=wall_collision, screen_width=820, screen_height=820,
        bullet_speed=240.0, targeted_probability=0.10,
        prediction_scale_min=0.65, prediction_scale_max=1.0, angular_noise=0.08,
    )
    layout = initialize_opening_bullets(
        300, 820, 820, 240.0, plane.position, plane.velocity,
        0.10, 0.65, 1.0, 0.08, rng, wall_collision=wall_collision,
    )
    rows = [
        [float(p[0]), float(p[1]), float(v[0]), float(v[1]), bool(flag)]
        for p, v, flag in zip(layout.positions, layout.velocities, layout.targeted)
    ]
    expected = deepcopy(rows)
    from barrage_rl.window_bullets import WindowBulletField
    field = WindowBulletField()
    field.append_batch(layout.positions, layout.velocities, layout.targeted)
    rows = field
    original_rows = list(rows)
    reference_rng = np.random.default_rng()
    reference_rng.bit_generator.state = deepcopy(rng.bit_generator.state)
    monkeypatch.setattr(Bullet, "LIST", rows)
    monkeypatch.setattr(Barrage, "KEY", True)
    monkeypatch.setattr(Barrage, "INVINCIBLE", False)
    monkeypatch.setattr(Barrage, "RNG", rng)
    monkeypatch.setattr(Barrage, "BULLET_FIELD_CONFIG", config)
    bullet = object.__new__(Bullet)
    for step in range(PHYSICS_FPS):
        if step % 17 == 0:
            # Preserve external edits through retained public row references.
            for index in (2, 17, 249):
                original_rows[index][0] = -2.0
                expected[index][0] = -2.0
        state = np.asarray(expected, dtype=np.float32)
        targeted = state[:, 4].astype(np.bool_)
        advance_bullet_field(
            state[:, :2], state[:, 2:4], targeted,
            plane.position, plane.velocity, reference_rng, PHYSICS_DT, config,
        )
        _reference_writeback(expected, state, targeted)
        bullet.update(plane, PHYSICS_DT)
        _check_rows_exact(rows, expected)
        assert rng.bit_generator.state == reference_rng.bit_generator.state
    assert Bullet.LIST is rows
    assert all(row is original for row, original in zip(rows, original_rows))
