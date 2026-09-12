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
        supporters = [max(lm._hypotheses[obj].evidence)
                      for lm in self.learning_modules
                      if lm.terminal_state == "match"
                      and lm.detected_object == obj
                      and obj in getattr(lm, "_hypotheses", {})]
        veto = self._expected_signal_veto(obj, supporters)
        if veto:
            logger.info(f"EXPECTED-SIGNAL VETO by {veto}: it was "
                        f"watching, it knows '{obj}', and the scene "
                        f"does not contain the tower '{obj}' predicts")
            return False
        logger.info(f"\n\nMONTY CONSENSUS MATCH: {n} senses agree on "
                    f"'{obj}'\n\n")
        return True

    # ------------------------------------------------------------------
    VETO_FRAC = 0.25   # the witness's tower must reach at least this
                       # fraction of the supporters' -- absence is
                       # RELATIVE ("it's not making a sound COMPARED
                       # to the sound that is present")

    def _expected_signal_veto(self, obj: str, supporter_ev: list):
        """Silence is not silence (Bryan, 2026-09-12): a sense never
        delivers nothing, it delivers a scene, and the candidate
        object predicts a specific tower standing above that scene's
        floor. If a FUNCTIONING sense (not dark, not numb) that has a
        LEARNED GRAPH for the candidate finds essentially no trace of
        it -- evidence far below what the agreeing senses are seeing
        (VETO_FRAC of their level; a startup transient's crumbs do
        not clear it) -- that is not abstention. It is testimony:
        'whatever this is, it is not the thing I would have heard.'
        The sax-mute differential, wired to a vote. Measured floor
        rationale: on a voiceless foil the ear's junk evidence for
        'mug' rides CARFAC's startup transient (2 stray percepts),
        small but positive -- an absolute zero test misses it; a
        relative one does not.

        A disabled sense (lit=False / numb=True) genuinely abstains:
        no data is different from data of no."""
        if not supporter_ev:
            return None
        floor = self.VETO_FRAC * (sum(supporter_ev) / len(supporter_ev))
        for i, lm in enumerate(self.learning_modules):
            if lm.terminal_state == "match" and lm.detected_object == obj:
                continue                       # a supporter, not a witness
            sms = getattr(self, "sensor_modules", None)
            sm = sms[i] if sms is not None and i < len(sms) else None
            if sm is not None and (getattr(sm, "lit", True) is False
                                   or getattr(sm, "numb", False)):
                continue                       # channel absent: abstains
            if obj not in lm.get_all_known_object_ids():
                continue                       # never learned it: no
                                               # prediction to violate
            hyp = getattr(lm, "_hypotheses", {}).get(obj)
            ev = (float(max(hyp.evidence))
                  if hyp is not None and len(hyp.evidence) else 0.0)
            if ev < floor:
                logger.info(
                    f"veto arithmetic: {lm.learning_module_id} sees "
                    f"{ev:.2f} for '{obj}' vs supporters' "
                    f"{[round(s, 1) for s in supporter_ev]} "
                    f"(floor {floor:.2f})"
                )
                return lm.learning_module_id   # looked where the tower
                                               # should stand: floor
        return None
