import concurrent
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List, Tuple
import concurrent.futures

import numpy as np


def log(x: float) -> float:
    with np.errstate(divide='ignore'):
        return np.log(x)


def best_path(mat):
    """
          mat: Output of neural network of shape LxNxC
    """
    classes = mat.argmax(-1)
    classes = classes.permute(1, 0)
    results = []
    for i, seq in enumerate(classes):
        seq = seq.unique_consecutive()
        seq = seq[seq != 10]
        result = ''.join([str(d.detach().cpu().item()) for d in seq])
        results.append(result)
    return results


@dataclass
class BeamEntry:
    """Information about one single beam at specific time-step."""
    pr_total: float = log(0)  # blank and non-blank
    pr_non_blank: float = log(0)  # non-blank
    pr_blank: float = log(0)  # blank
    labeling: tuple = ()  # beam-labeling


class BeamList:
    """Information about all beams at specific time-step."""

    def __init__(self) -> None:
        self.entries = defaultdict(BeamEntry)

    def sort_labelings(self) -> List[Tuple[int]]:
        """Return beam-labelings, sorted by probability."""
        beams = self.entries.values()
        sorted_beams = sorted(beams, reverse=True, key=lambda x: x.pr_total)
        return [x.labeling for x in sorted_beams]


def beam_search(mat: np.ndarray, beam_width: int):
    """Beam search decoder.
    inspired by: https://github.com/githubharald/CTCDecoder
    See the paper of Hwang et al. and the paper of Graves et al.

    Args:
        mat: Output of neural network of shape TxC.
    Returns:
        The decoded text.
        :param beam_width:
    """

    blank_idx = 10
    max_T, max_C = mat.shape

    # initialise beam state
    last = BeamList()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].pr_blank = log(1)
    last.entries[labeling].pr_total = log(1)

    # go over all time-steps
    for t in range(max_T):
        curr = BeamList()

        # get beam-labelings of best beams
        best_labelings = last.sort_labelings()[:beam_width]

        # go over best beams
        for labeling in best_labelings:

            # probability of paths ending with a non-blank
            pr_non_blank = log(0)
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                pr_non_blank = last.entries[labeling].pr_non_blank + mat[t, labeling[-1]]

            # probability of paths ending with a blank
            pr_blank = last.entries[labeling].pr_total + mat[t, blank_idx]

            # fill in data for current beam
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].pr_non_blank = np.logaddexp(curr.entries[labeling].pr_non_blank, pr_non_blank)
            curr.entries[labeling].pr_blank = np.logaddexp(curr.entries[labeling].pr_blank, pr_blank)
            curr.entries[labeling].pr_total = np.logaddexp(curr.entries[labeling].pr_total,
                                                           np.logaddexp(pr_blank, pr_non_blank))
            # extend current beam-labeling
            for c in range(max_C - 1):
                # add new char to current beam-labeling
                new_labeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    pr_non_blank = last.entries[labeling].pr_blank + mat[t, c]
                else:
                    pr_non_blank = last.entries[labeling].pr_total + mat[t, c]

                # fill in data
                curr.entries[new_labeling].labeling = new_labeling
                curr.entries[new_labeling].pr_non_blank = np.logaddexp(curr.entries[new_labeling].pr_non_blank,
                                                                       pr_non_blank)
                curr.entries[new_labeling].pr_total = np.logaddexp(curr.entries[new_labeling].pr_total, pr_non_blank)

        # set new beam state
        last = curr

    # sort by probability
    best_labeling = last.sort_labelings()[0]  # get most probable labeling
    result = ''.join([str(x) for x in best_labeling])
    return result


def beam_search_batch(mat, beam_width):
    mat = mat.permute(1, 0, 2).detach().cpu().numpy()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda m: beam_search(m, beam_width), mat))

    return results
