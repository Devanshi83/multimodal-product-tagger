import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_precision_at_k_range():
    from evaluation.metrics import precision_at_k
    labels = (np.random.rand(50, 10) > 0.7).astype(float)
    scores = np.random.rand(50, 10)
    result = precision_at_k(labels, scores, k=1)
    assert 0.0 <= result <= 1.0

def test_precision_at_k_perfect():
    from evaluation.metrics import precision_at_k
    labels = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
    scores = np.array([[0.9,0.1,0.1],[0.1,0.9,0.1],[0.1,0.1,0.9]], dtype=float)
    assert precision_at_k(labels, scores, k=1) == 1.0

def test_compute_all_metrics_keys():
    from evaluation.metrics import compute_all_metrics
    y_true = (np.random.rand(30, 5) > 0.5).astype(float)
    y_prob = np.random.rand(30, 5).astype(float)
    result = compute_all_metrics(y_true, y_prob, threshold=0.5)
    assert "mAP" in result
    assert "hamming_loss" in result
    assert "f1_micro" in result
    assert "f1_macro" in result
    assert "precision_at_1" in result

def test_compute_all_metrics_ranges():
    from evaluation.metrics import compute_all_metrics
    y_true = (np.random.rand(30, 5) > 0.5).astype(float)
    y_prob = np.random.rand(30, 5).astype(float)
    result = compute_all_metrics(y_true, y_prob)
    assert 0.0 <= result["mAP"] <= 1.0
    assert 0.0 <= result["hamming_loss"] <= 1.0
    assert 0.0 <= result["f1_micro"] <= 1.0
