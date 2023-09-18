from Levenshtein import distance as levenshtein


def calculate_cer_for_batch(target_lengths, target, decoded):
    cer = 0
    for l, t, d in zip(target_lengths, target, decoded):
        ref = ''.join([str(x.item()) for x in t[:l].detach().cpu()])
        cer += levenshtein(d, ref) / l
    return cer / len(target_lengths)
