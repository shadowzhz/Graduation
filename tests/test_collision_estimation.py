"""Wall-only estimator tests; truth is used for assertions, never filter inputs."""

import math

import numpy as np

from air_hockey import core_config as core
from air_hockey.estimation import KalmanFilter
from air_hockey.physics import StoneMotion, apply_friction_velocity
from air_hockey.simulation import MotionSimulator


def _filter(model, state):
    estimator = KalmanFilter(motion_model=model)
    estimator.update(state[0], state[1], 0.0)
    estimator._x[:] = state
    estimator._friction_started = True
    return estimator


def test_collision_model_matches_friction_without_wall_contact():
    state = (300.0, 400.0, 80.0, -60.0)
    friction = _filter("friction", state)
    collision = _filter("collision_aware", state)
    for index in range(1, 31):
        timestamp = index / 60.0
        for estimator in (friction, collision):
            if index % 3:
                estimator.predict(timestamp)
            else:
                estimator.update(300.0 + index, 400.0 - index, timestamp)
        np.testing.assert_array_equal(collision.state_vector, friction.state_vector)
        np.testing.assert_array_equal(collision._P, friction._P)


def test_collision_prediction_reflects_all_four_wall_normals():
    left = core.RINK_LEFT + core.STONE_RADIUS
    right = core.RINK_RIGHT - core.STONE_RADIUS
    top = core.RINK_TOP + core.STONE_RADIUS
    bottom = core.RINK_BOTTOM - core.STONE_RADIUS
    cases = (
        ((left + 1.0, 400.0, -200.0, 30.0), 0, left),
        ((right - 1.0, 400.0, 200.0, 30.0), 0, right),
        ((left + 50.0, top + 1.0, 30.0, -200.0), 1, top),
        ((left + 50.0, bottom - 1.0, 30.0, 200.0), 1, bottom),
    )
    dt = 0.01
    for initial, axis, limit in cases:
        estimator = _filter("collision_aware", initial)
        state = estimator.predict(dt)
        reflected = list(initial[2:])
        reflected[axis] *= -core.WALL_RESTITUTION
        expected_velocity = apply_friction_velocity(*reflected, dt)
        assert (state.x, state.y)[axis] == limit
        np.testing.assert_allclose((state.vx, state.vy), expected_velocity, atol=1e-12, rtol=0.0)
        assert (state.vx, state.vy)[axis] * initial[axis + 2] < 0.0
        assert (state.vx, state.vy)[1 - axis] * initial[3 - axis] > 0.0


def test_collision_prediction_does_not_reflect_away_from_wall():
    left = core.RINK_LEFT + core.STONE_RADIUS
    initial = (left + 0.1, 400.0, 100.0, 20.0)
    collision = _filter("collision_aware", initial)
    friction = _filter("friction", initial)
    collision.predict(0.01)
    friction.predict(0.01)
    np.testing.assert_array_equal(collision.state_vector, friction.state_vector)


def test_noisy_outside_measurement_does_not_trigger_velocity_flip():
    left = core.RINK_LEFT + core.STONE_RADIUS
    initial = (left + 2.0, 400.0, -80.0, 0.0)
    collision = _filter("collision_aware", initial)
    friction = _filter("friction", initial)
    collision._P = friction._P = np.eye(4)
    # A noisy sample is outside the wall, but the process prediction has not contacted it.
    collision.update(left - 2.0, 400.0, 0.01)
    friction.update(left - 2.0, 400.0, 0.01)
    np.testing.assert_array_equal(collision.state_vector, friction.state_vector)
    assert collision.state_vector[2] < 0.0
    # An outward-to-inward noisy displacement while moving away must not create another bounce.
    away = _filter("collision_aware", (left + 1.0, 400.0, 80.0, 0.0))
    away._P = np.eye(4)
    assert away.update(left - 2.0, 400.0, 0.01).vx > 0.0
    assert away.predict(0.02).vx > 0.0


def test_collision_predict_frames_and_update_do_not_duplicate_bounce():
    left = core.RINK_LEFT + core.STONE_RADIUS
    initial = (left + 1.0, 400.0, -200.0, 20.0)
    estimator = _filter("collision_aware", initial)
    truth = StoneMotion(x=initial[0], y=initial[1], vx=initial[2], vy=initial[3])
    for index in range(1, 4):
        truth.step(0.01)
        state = (estimator.predict(index * 0.01) if index < 3 else
                 estimator.update(truth.x, truth.y, index * 0.01))
        np.testing.assert_allclose((state.x, state.y, state.vx, state.vy),
                                   (truth.x, truth.y, truth.vx, truth.vy), atol=1e-10, rtol=0.0)
        assert state.vx > 0.0
    before = estimator.state_vector
    estimator.predict(0.03)
    np.testing.assert_array_equal(estimator.state_vector, before)


def test_collision_prediction_keeps_goal_mouth_open_and_ignores_posts():
    initial = (core.RINK_CENTER_X, core.RINK_TOP + core.STONE_RADIUS + 1.0, 0.0, -200.0)
    collision = _filter("collision_aware", initial)
    friction = _filter("friction", initial)
    collision.predict(0.02)
    friction.predict(0.02)
    np.testing.assert_array_equal(collision.state_vector, friction.state_vector)
    post_x, post_y = core.GOAL_POSTS[0]
    initial = (post_x + core.STONE_RADIUS + 1.0, post_y + 10.0, 0.0, -80.0)
    collision = _filter("collision_aware", initial)
    friction = _filter("friction", initial)
    collision.predict(0.01)
    friction.predict(0.01)
    np.testing.assert_array_equal(collision.state_vector, friction.state_vector)
    truth = StoneMotion(x=initial[0], y=initial[1], vx=initial[2], vy=initial[3])
    assert truth.step(0.01)
    assert truth.vx != 0.0


def test_wall_prediction_covariance_matches_finite_difference():
    left = core.RINK_LEFT + core.STONE_RADIUS
    initial = np.array([left + 1.0, 400.0, -200.0, 30.0])
    dt, epsilon = 0.01, 1e-4
    def mean(state):
        estimator = _filter("collision_aware", state)
        estimator._advance(dt)
        return estimator.state_vector
    jacobian = np.column_stack([
        (mean(initial + np.eye(4)[axis] * epsilon) - mean(initial - np.eye(4)[axis] * epsilon)) / (2 * epsilon)
        for axis in range(4)
    ])
    estimator = _filter("collision_aware", initial)
    estimator._P = np.diag([2.0, 3.0, 5.0, 7.0])
    expected = jacobian @ estimator._P @ jacobian.T + estimator._process_noise(dt)
    estimator._advance(dt)
    np.testing.assert_allclose(estimator._P, expected, rtol=1e-5, atol=1e-7)
    assert np.linalg.eigvalsh(estimator._P).min() >= -1e-10


def test_seed_51_wall_direction_recovers_on_contact_without_truth_input():
    simulator = MotionSimulator.random(seed=51, position_noise=0.0)
    results = simulator.simulate_estimators(("friction", "collision_aware"), predict_step=55)
    collision, friction = results["collision_aware"], results["friction"]
    bounce = next(index for index in range(1, len(collision.true_velocity_trajectory))
                  if collision.true_velocity_trajectory[index - 1][0] < 0 < collision.true_velocity_trajectory[index][0])
    assert collision.filtered_velocity_trajectory[bounce][0] > 0.0
    assert collision.filtered_state.vx > 0.0 > friction.filtered_state.vx
    assert collision.endpoint_error() < friction.endpoint_error()
    assert math.dist((collision.filtered_state.vx, collision.filtered_state.vy),
                     collision.true_velocity_trajectory[55]) < 0.01
