"""Novelty by disagreement: consensus termination for the federation.

Bryan's rule, stated on 2026-09-12 after the foil test: novelty is
not a per-sense threshold, it is DISAGREEMENT BETWEEN SENSES. The
stock criterion counts confident learning modules without asking
what they are confident ABOUT -- measured: a silent bowl read
mug-by-eye and banana-by-touch, and AnyLMsMatch(count=2) called
that a match. Two witnesses naming different suspects is not a
conviction.

MontyConsensus overrides one decision (check_terminal_conditions)
and changes nothing else:

    match  <=>  at least N matched LMs name the SAME object
                AND no matched LM names a different one.

Confident dissent blocks the verdict. A lone confident sense
(n=1 < N) is an opinion, not an identification. Senses that stay
silent (the ear on a voiceless foil) neither confirm nor veto --
though their silence keeps N unreachable at N=3, which is the
strictest honest setting: full cross-modal confirmation, where a
mug-shape that does not hum is NOT a mug.

N is read from the configured match_criterion's count, so the
existing yaml/CLI plumbing (experiment.config.match_criterion.
count=2) sets the consensus size. No Monty code touched -- this is
a subclass in our package, pointed at via monty_class in the
experiment config, the same socket every sensor used.
"""
from __future__ import annotations

import logging
from collections import Counter

from tbp.monty.frameworks.models.evidence_matching.model import (
    MontyForEvidenceGraphMatching,
)

logger = logging.getLogger(__name__)


class MontyConsensus(MontyForEvidenceGraphMatching):
    def check_terminal_conditions(self) -> bool:
        if all(lm.terminal_state == "no_match"
               for lm in self.learning_modules):
            return True
        if not self.exceeded_min_steps:
            return False

        named = []
        for lm in self.learning_modules:
            lm.update_terminal_condition()
            if lm.terminal_state == "match" and lm.detected_object:
                named.append((lm.learning_module_id, lm.detected_object))
        if not named:
            return False

        counts = Counter(obj for _, obj in named)
        obj, n = counts.most_common(1)[0]
        dissent = len(named) - n
        need = getattr(self._match_criterion, "_count", 2)
        if dissent > 0:
            logger.info(
                f"CONSENSUS BLOCKED: {dict(counts)} -- confident "
                f"disagreement, no verdict"
            )
            return False
        if n < need:
            logger.info(
                f"CONSENSUS PENDING: {n}/{need} name '{obj}' -- a lone "
                f"opinion is not an identification"
            )
            return False
        logger.info(f"\n\nMONTY CONSENSUS MATCH: {n} senses agree on "
                    f"'{obj}'\n\n")
        return True
