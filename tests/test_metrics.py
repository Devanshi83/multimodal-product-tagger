import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def test_hamming_loss_perfect():
    from evaluation.metrics import hamming_loss
    y = np.array([[1,0,1],[0,1,0]])
    assert hamming_loss(y, y) == 0.0

def test_hamming_loss_all_wrong():
    from evaluation.metrics import hamming_loss
    y_true = np.array([[1,0,1],[0,1,0]])
    y_pred = np.array([[0,1,0],[1,0,1]])
    assert hamming_loss(y_true, y_pred) == 1.0

def test_precision_at_k_range():
    from evaluation.metrics import precision_at_k
    labels = (np.random.rand(50,10) > 0.7).astype(float)
    scores = np.random.rand(50,10)
    assert 0.0 <= precision_at_k(labels, scores, k=1) <= 1.0

def test_map_range():
    from evaluation.metrics import mean_average_precision
    labels = (np.random.rand(50,8) > 0.6).astype(float)
    scores = np.random.rand(50,8)
    assert 0.0 <= mean_average_precision(labels, scores) <= 1.0
