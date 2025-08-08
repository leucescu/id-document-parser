import torch
from training.metrics import calculate_token_accuracy, calculate_sequence_accuracy, calculate_average_edit_distance

def test_calculate_token_accuracy_basic():
    pred_ids = torch.tensor([[1, 2, 3, 0]])
    true_ids = torch.tensor([[1, 2, 4, 0]])
    pad_token_id = 0
    acc = calculate_token_accuracy(pred_ids, true_ids, pad_token_id)
    # 2 correct out of 3 non-pad tokens
    assert abs(acc - (2/3)) < 1e-6

def test_calculate_token_accuracy_all_padding():
    pred_ids = torch.tensor([[0, 0, 0]])
    true_ids = torch.tensor([[0, 0, 0]])
    pad_token_id = 0
    acc = calculate_token_accuracy(pred_ids, true_ids, pad_token_id)
    # No valid tokens to compare, accuracy should be 0 or handled gracefully
    assert acc == 0.0

def test_calculate_token_accuracy_empty_tensors():
    pred_ids = torch.empty((0,0), dtype=torch.long)
    true_ids = torch.empty((0,0), dtype=torch.long)
    pad_token_id = 0
    acc = calculate_token_accuracy(pred_ids, true_ids, pad_token_id)
    assert acc == 0.0

def test_calculate_sequence_accuracy_basic():
    preds = ["abc", "def", "ghi"]
    trues = ["abc", "xyz", "ghi"]
    acc = calculate_sequence_accuracy(preds, trues)
    assert acc == 2/3

def test_calculate_sequence_accuracy_empty_lists():
    acc = calculate_sequence_accuracy([], [])
    assert acc == 0.0

def test_calculate_average_edit_distance_basic():
    preds = ["kitten", "flaw"]
    trues = ["sitting", "lawn"]
    dist = calculate_average_edit_distance(preds, trues)
    expected = (3 + 2) / 2
    assert abs(dist - expected) < 1e-6

def test_calculate_average_edit_distance_empty_lists():
    dist = calculate_average_edit_distance([], [])
    assert dist == 0.0