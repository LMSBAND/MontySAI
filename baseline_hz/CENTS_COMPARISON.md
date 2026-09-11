# The unit surgery, scored (2026-09-10, late)

Code: 509cebc (pitch_semis, width-matched 136 cents) vs this frozen
baseline. Every experiment re-run; verdict / most-likely / steps
compared row by row:

| experiment | Hz era | cents era | |
|---|---|---|---|
| synthetic control (mug) | correct→mug@26 | correct→mug@26 | SAME |
| synthetic control (banana) | correct→banana@21 | correct→banana@21 | SAME |
| synthetic transposed | correct→mug@26 | correct→mug@26 | SAME |
| riffs control (mug) | correct→mug@21 | correct→mug@21 | SAME |
| riffs control (banana) | correct→banana@21 | correct→banana@21 | SAME |
| riffs C minor (both) | correct@21 | correct@21 | SAME |
| riffs major (mug) | no_match→mug@21 | no_match→mug@21 | SAME |
| ibz AM | correct→mug@21 | correct→mug@21 | SAME |
| ibz CM | confused→banana@21 | confused→banana@21 | SAME |
| ibz CMAJ | correct→mug@21 | correct→mug@21 | SAME |

**Twelve verdicts, twelve identical.** The Hz bug was real but benign
at verdict level: identity rides on location + timbre + place; the
pitch feature was near-vestigial in the lived register. All findings
stand on the cents-true ear.

Open on purpose: mode-in-residual needs the WIDTH limb (classifier
tolerance sweep, analysis-only). Different night, one limb at a time.
