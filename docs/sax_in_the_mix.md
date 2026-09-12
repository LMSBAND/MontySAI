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


## The pitch-null (same evening, the reviewer's five-minute rerun)

Null that keeps the moment and moves the lag (a semitone off, both
directions): band alignment survives, sax content leaves. Result:
MUTE passes it too (right-lag wins 77/61/61% without any sax) --
because the sax plays chord tones and THE BAND VOICES THE SAME
HARMONY. A semitone off the sax's note is a semitone off everyone's
note. Moment+pitch probing detects the harmony, not the instrument.

The confound now has two measured layers: shift time and you lose
the song; shift pitch and you leave the chord. The sax hides inside
the harmony's shadow. FULL still exceeds MUTE at every row (20.3 vs
16.2, 4.0 vs 1.6, 2.6 vs 1.9) -- the increment is real but smaller
than the harmony floor, so one-file detection requires a
source-specific feature at the probed (moment, lag): timbre-at-lag
or per-row structure. Queued next: the feedback experiment -- the
LM biasing the sax's channels/lags top-down, prediction: the 60-66%
whisper becomes a voice.


## Closing the loop (2026-09-11, night): attention, and its price

Bryan's directive: feedback from the model to the sense. Three
stages, each measured:

1. **Place-efferent (the MOC pathway)**: suppress the bass's cochlear
   places (OHC gain x0.05 below 170 Hz -- the lesion knob under
   different politics; the wire real brains use). Suppression landed
   (in-band NAP -30%, mid-band untouched) and CHANGED NOTHING:
   the bass's periodicity survives via harmonics at unsuppressed
   places, which moreover share places with the sax. Finding: pitch
   is not at a place; place-attention cannot unmask a periodicity.

2. **Lag-window attention (readout level)**: the tracker searches
   only the target register's lag band. Percept-level fit to the sax
   memory: 16% -> 71% in the FULL mix. The whisper became a shout.
   But MUTE jumped too (14% -> 68%): the harmony's shadow, again --
   location cannot tell a sax from the chord it is playing.

3. **The verdicts**: FULL attending -> leaning sax at timeout
   (blind: confidently 'band'). MUTE attending -> CONFIDENTLY SAX in
   25 steps. The false-positive trap that held 9/9 unbidden fired
   the moment the system was told where to look. **Attention without
   a discriminative feature is confirmation bias, measured.** The
   current timbre tolerance (0.4 log-ratio) cannot separate comping
   midrange from horn.

The loop is closed and its first lesson is the biological one:
top-down expectation manufactures percepts unless a source-specific
feature polices it. Next component: not more attention -- a sharper
timbre (or per-row structure) at the attended lag, tight enough to
refuse the shadow.


## The timbre diagnostic (what attention needs, measured)

Can the current 4-number timbre police attention? NO, and the reason
is structural: per-element sax-vs-shadow gaps are 0.04/0.03/0.04/0.27
(tolerance 0.40 each) -- three of four numbers are identical between
horn and comping shadow. And the sax matches ITS OWN centroid only
40% of the time: a sax's harmonic recipe changes with the note it
plays, so within-source variance swamps the between-source gap. No
tolerance anywhere separates FULL from MUTE (best gap +4.8 points).

Feature upgrade spec, from the data: (a) higher harmonics (h6-h10 --
only h5 carried any signal; horns keep structure up high, plucked
comping decays); (b) odd/even harmonic ratio (reed signature);
(c) SUSTAIN SLOPE at the attended lag across frames -- a horn holds
and swells where comping chords decay; likely the sharpest single
cue and it is temporal, which this whole architecture is built to
honor; (d) pitch-conditioning (compare timbre against the memory
node's own register, not a global centroid -- the node-local timbre
the graph already stores, which the acceptance test above ignored).
