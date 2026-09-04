import numpy as np
import jax, jax.numpy as jnp


def _run(fly_sex, fly_valid, prompt_on, dist):
    from jarvis_jax.train.matching import assign_slots
    a, t = assign_slots(jnp.asarray(fly_sex, jnp.int8), jnp.asarray(fly_valid), jnp.asarray(prompt_on),
                        jnp.asarray(dist, jnp.float32))
    return np.asarray(a).tolist(), np.asarray(t).tolist()


def test_single_female_unprompted_goes_to_female_slot():
    a, t = _run([[0, -1]], [[True, False]], [False], [[0.0, 0.0]])
    assert a == [[1, -1]] and t == [[False, True, False, False]]


def test_single_male_prompted_goes_to_slot0_and_male_slot_is_empty():
    a, t = _run([[1, -1]], [[True, False]], [True], [[0.0, 0.0]])
    assert a == [[0, -1]] and t == [[True, False, False, False]]


def test_mixed_pair_unprompted_and_prompted():
    a, t = _run([[0, 1]], [[True, True]], [False], [[0.0, 30.0]])
    assert a == [[1, 2]] and t == [[False, True, True, False]]
    a, t = _run([[0, 1]], [[True, True]], [True], [[0.0, 30.0]])
    assert a == [[0, 2]] and t == [[True, False, True, False]]


def test_same_sex_pair_nearer_fly_takes_typed_slot():
    # host is fly 0 but the OTHER female is nearer the ROI origin: order is host first, so the
    # host still takes slot 1 (host-first rule beats distance); the other goes to slot 3
    a, t = _run([[0, 0]], [[True, True]], [False], [[10.0, 2.0]])
    assert a == [[1, 3]] and t == [[False, True, False, True]]
    # prompted: host -> 0, the other female -> the (free) female slot 1
    a, _ = _run([[0, 0]], [[True, True]], [True], [[10.0, 2.0]])
    assert a == [[0, 1]]


def test_unknown_sex_goes_to_other_slot_and_invalid_is_minus_one():
    a, t = _run([[-1, 1]], [[True, False]], [False], [[0.0, 0.0]])
    assert a == [[3, -1]] and t == [[False, False, False, True]]


def test_batched_and_jittable():
    from jarvis_jax.train.matching import assign_slots
    f = jax.jit(assign_slots)
    a, t = f(jnp.asarray([[0, 1], [1, -1]], jnp.int8), jnp.asarray([[True, True], [True, False]]),
             jnp.asarray([False, True]), jnp.zeros((2, 2), jnp.float32))
    assert np.asarray(a).tolist() == [[1, 2], [0, -1]]
    assert np.asarray(t).shape == (2, 4)


def test_slot_ignore_codes():
    from jarvis_jax.train.matching import slot_ignore
    ig = np.asarray(slot_ignore(jnp.asarray([-1, 0, 1, 2], jnp.int8)))
    assert ig.tolist() == [[False, False, False, False],
                           [False, True, False, True],
                           [False, False, True, True],
                           [False, True, True, True]]
