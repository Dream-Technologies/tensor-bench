import math

import numpy as np


def softmax(z):
    z_exp = np.exp(z)
    return z_exp / np.sum(z_exp, axis=1, keepdims=True)


def cross_entropy_with_softmax(y_true, z):
    probs = softmax(z)
    loss = -np.mean(np.sum(y_true * np.log(probs), axis=1))
    return loss, probs


def run_tests():
    print("Starting ML Engineering tests...\n")

    # ---------------------------------------------------------
    # TEST 0
    # ---------------------------------------------------------
    z_basic = np.array([[0.6, 0.4]])
    y_true_basic = np.array([[1.0, 0.0]])

    loss, probs = cross_entropy_with_softmax(y_true_basic, z_basic)

    expected_probs = np.array([[0.549834, 0.450166]])
    expected_loss = -math.log(0.549834)

    assert np.allclose(probs, expected_probs), f"FAIL: Expected probs {expected_probs}, got {probs}"
    assert math.isclose(loss, expected_loss, rel_tol=1e-5), f"FAIL: Expected loss {expected_loss}, got {loss}"
    print("✓ Test 0 Passed.")

    # ---------------------------------------------------------
    # TEST 1
    # ---------------------------------------------------------
    z_large = np.array([[1000.0, 1000.0]])
    y_true_large = np.array([[1.0, 0.0]])

    loss, probs = cross_entropy_with_softmax(y_true_large, z_large)

    assert not np.isnan(probs).any(), "FAIL: Softmax returned NaNs."
    assert np.allclose(probs, [[0.5, 0.5]]), f"FAIL: Expected probs [0.5, 0.5], got {probs}"
    print("✓ Test 1 Passed.")

    # ---------------------------------------------------------
    # TEST 2
    # ---------------------------------------------------------
    z_extreme = np.array([[-1000.0, 1000.0]])
    y_true_extreme = np.array([[1.0, 0.0]])

    loss, probs = cross_entropy_with_softmax(y_true_extreme, z_extreme)

    assert not np.isinf(loss), "FAIL: Loss returned inf."
    assert not np.isnan(loss), "FAIL: Loss returned NaN."

    # The true class gets ~0 probability, so the loss should be large.
    # Exact value depends on the fix (clipping eps vs. log-sum-exp), so only check it's big.
    assert loss > 10, f"FAIL: Expected a large loss for a confidently wrong prediction, got {loss}"
    print("✓ Test 2 Passed.")


if __name__ == "__main__":
    run_tests()
