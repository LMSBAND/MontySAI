# There is a sax in here: the detection campaign (2026-09-11)

Design (Bryan's): three mixes, three stems each -- SAXSOLO, FULL,
and SAXMUTE (the band plays on with the sax removed: the built-in
false-positive trap). The claim under test is DETECTION -- "there is
a sax in here" -- not classification. Memory per mix: the sax (SOLO)
and the band-without-it (MUTE) as the two objects, so the alternative
hypothesis is the band itself.

## Results, four levels deep

1. **Verdicts** (winner-take-all percepts): SOLO -> sax 3/3;
   MUTE -> band 3/3 (nine chances to hallucinate a sax, zero taken:
   perfect specificity); FULL -> band 3/3 (zero sensitivity). The
   sanity stats predicted it: FULL and MUTE percept medians are
   identical (~75-82 Hz) -- the tracker rides the bass; the sax
   (208-245 Hz) almost never gets the microphone.

2. **Residual fit** (do FULL's percepts fit the sax memory more than
   MUTE's?): +4.7 / +3.1 / +0.0 points. Nearly blind -- unlike the
   mode study, the information never entered the percept stream.

3. **Memory-guided differential** (sample-aligned stems: probe FULL
   and MUTE at the sax's own (moment, lag) track from SOLO): FULL
   beats MUTE at 66% / 61% / 60% of ~1,070 moments per mix. THE SAX
   IS IN THE IMAGE; the extractor is what's blind.

4. **Self-contained detector** (aligned vs time-shifted track within
   one file): defeated -- MUTE also rewards alignment (+7.2 on mix1)
   because THE BAND IS PLAYING THE SAME SONG. In music, everything is
   correlated with everything; timing alone cannot isolate a source
   from an ensemble that agrees with it.

## What it means

Three problems now converge on one missing mechanism -- chord
quality, the cocktail party, and sax-in-the-mix all fail at the same
place: winner-take-all percept extraction discards minority
periodicities before anything downstream can use them. Per-row
runner-up extraction is the gate. The self-correlation result adds a
second requirement: breaking the song-confound needs a
source-specific feature at the probed lag (timbre-at-lag), not
timing alone.
