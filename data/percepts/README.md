# Percept logs

Cached percept streams from the analysis runs, as numpy arrays.

| file | contents |
|---|---|
| `ibz_{AM,CM,CMAJ}_IBANEZ.npy` | [n, 3] = (x, y, pitch_hz) per passed percept, first 30 s of each solid-body take |
| `cm_stream.npy` | hollow-body C-minor take, same format |
| `transposed_locs.npy` | [n, 3] locations of the synthetic fifth-up siren stream |
| `maj_anatomy.npy` | hollow-body major take: [x, y, pitch, dist-to-nearest-node, pitch-diff, fits] |

Coordinates: x = log2(lag/1 ms) x 0.1 m, y = channel-centroid x 0.004 m.
Regenerate any of these with `audiomonty.anatomy.percept_stream(path)`
against the takes in `takes/`.
