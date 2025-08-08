from Levenshtein import distance as levenshtein_distance

def calculate_token_accuracy(pred_ids, true_ids, pad_token_id):
    mask = true_ids != pad_token_id
    total = mask.sum().item()
    if total == 0:
        return 0.0
    correct = ((pred_ids == true_ids) & mask).sum().item()
    return correct / total

def calculate_sequence_accuracy(pred_texts, true_texts):
    if len(pred_texts) == 0:
        return 0.0
    correct = sum(p == t for p, t in zip(pred_texts, true_texts))
    return correct / len(pred_texts)

def calculate_average_edit_distance(pred_texts, true_texts):
    if len(pred_texts) == 0:
        return 0.0
    distances = [levenshtein_distance(p, t) for p, t in zip(pred_texts, true_texts)]
    return sum(distances) / len(distances)