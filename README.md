# cellmat
Some utils for matching neurons across different recordings

## Methods for matching cells
Let's say we are comparing unit A from day 1 to unit B from day 2.
- waveform nearest neighbor distance
  - project the spikes from both A and B to PC space; keep the dimensionality small
  - for each spike of A, compute the fraction of $m$ nearest neighbors that belongs to B
  - divide by total number of spikes in A; this is the "nearest neighbor distance"
  - if this exceeds some threshold (e.g. 0.2), then A and B are sufficiently close and are considered a match
- template similarity
  - compute the cosine simliarity between templates (average waveforms) of A and B
  - if this exceeds some threshold (e.g. 0.9) then A and B are sufficiently similar and are considered a match
  - for high channel count probes, restrict to channels that carry signal 
    - define channels with signal as channels that are some radius from the max channel (channel with largest peak to peak template)
    - take the union of channels with signal for A and B
  - ref: Lee et al. 2021
- autocorrelogram similarity
  - compute autocorrelogram for both A and B within some window (e.g. 50 ms) and fixed bin size (e.g. 2 ms)
  - compute cosine similarity between the autocorrelograms
  - if this exceeds some threshold (e.g. 0.9) then A and B are sufficiently similar and are considered a match
  - ref: Schoonover et al. 2021
- firing rate
  - two neurons with similar firing rates are a match
  - similarity in firing rate may be defined as the differnence being within some fraction of original firing rate
- tuning properties
  - define response vector of A and B as the peak firing rate to a set of stimuli and compute their cosine similarity
  - if this exceeds some threshold (e.g. 0.9) then A and B are sufficiently similar and are considered a match