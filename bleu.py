from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def calculate_bleu(references, hypotheses, max_n=4):
    """
    Calculate BLEU score for a batch of sequences.
    :param references: List of reference sequences (ground truth)
    :param hypotheses: List of hypothesis sequences (model predictions)
    :param max_n: Maximum n-gram to consider (default is 4 for BLEU-4)
    :return: BLEU score
    """
    weights = [1 / max_n] * max_n
    bleu_scores = []
    smoothing_function = SmoothingFunction().method1

    # for i, (ref, hyp) in enumerate(zip(references, hypotheses)):
    for ref, hyp in zip(references, hypotheses):
        # if i < 2:
        #     print(f"Reference {i}: {ref}")
        #     print(f"Hypothesis {i}: {hyp}")
        bleu_scores.append(
            sentence_bleu(
                [ref], hyp, weights=weights, smoothing_function=smoothing_function
            )
        )

    return sum(bleu_scores) / len(bleu_scores)
