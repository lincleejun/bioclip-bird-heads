# Beak/eye and flight heads

Two small heads on the frozen int8 BioCLIP 2.5 image tower of
[bioclip-bird](https://github.com/lincleejun/bioclip-bird). Both read the same thing: the 256 patch tokens of
transformer block 15 (0-based, of 32) that the tower computes anyway while it names the bird, so one forward
pass gives the species, the beak and eye points, and P(flying).

Every number below is copied from `runs/parts-1/report.json`, `runs/flight-1/report.json` or the project's
working notes (marked *notes*); the run directories are not published.

## What the heads read

- Input: a bird crop framed the way gardener frames it for the species encoder (the detector's box plus 10 %
  on each side, made square), then the tower's own preprocessing: shortest side to 224 px, centre crop
  224 x 224, CLIP mean/std.
- Tokens: block 15's output, 16 x 16 = 256 patch tokens (14 px patches), 1,280 wide; the class token is
  dropped. The tower ran in float16 during training.
- Why block 15: on 2,000 images of the CUB train split (1,500 to fit, 500 to compare), blocks 11, 15, 19, 23, 27
  and 31 were tried; block 15 localised best and the last block clearly worse (*notes*; the probe's output was
  not kept). For flight, a linear layer on the final image vector (the one used to name the species) scored
  81-88 % on unseen species and called perched gulls and egrets "flying": it learns which species tend to be
  photographed in flight. Block 15 still carries pose (*notes*).

## Beak and eyes (`parts.py`, `parts.safetensors`, 1.3 MB)

**Shape.** Per patch token: LayerNorm(1280) -> Linear(1280, 256) -> GELU -> Linear(256, 9), i.e. for each of
beak, left eye, right eye a "the part is in this patch" logit and an (x, y) offset inside the patch (sigmoid).
Per image: Linear(256, 3) on the mean of the hidden tokens gives "is this part visible". A point is the
arg-max patch plus its offset, in the 224 px frame (`examples/identify.py` in bioclip-bird maps it back to the
crop's pixels). Left and right are the bird's own, as CUB labels them.

**Data.** CUB-200-2011 (11,788 images, 200 species), its official train/test split; each image's beak, left eye
and right eye points and the bird's box. A point that the framing cuts off counts as not visible.
CUB-200-2011 is for non-commercial research and educational use only.

**Training.** 5,994 train images, 30 epochs, batch 64, AdamW (lr 1e-3, weight decay 0.01, one-cycle), seed 0.
Loss: cross-entropy over the 256 patches + L1 on the in-patch offset (visible parts only) + binary
cross-entropy on visibility.

```
./run ../bioclip-bird-heads/parts.py probe                           # block choice, train split only
./run ../bioclip-bird-heads/parts.py train --layer 15 --out runs/parts-1
```

**Test.** The 5,794 test images, scored once. Error = distance to the true point as a share of the 224 px
frame's side. "Eye" is what a culling tool needs: the more confident of the two predicted eyes against the
nearest visible true eye (CUB's left/right labels are not reliable).

| CUB test | n | median error | within 5 % | within 10 % | visible judged right |
|---|---|---|---|---|---|
| beak | 5,572 | 1.14 % | 97.8 % | 99.4 % | 97.7 % |
| eye | 5,561 | 0.92 % | 98.2 % | 99.2 % | 95.4 % (either eye) |
| left eye / right eye | 2,876 / 2,927 | 0.93 % / 0.92 % | 98.2 % / 98.2 % | 99.2 % / 99.0 % | 94.2 % / 95.1 % |

**Checks against fooling ourselves.**

- Baseline "always the mean position": beak median 30.9 %, within 5 % 0.8 %; eye 23.8 %, 3.9 %.
- Detector-like boxes: every side of the box moved at random by up to 15 % of its size: beak median 1.14 %,
  98.0 % within 5 %; eye 0.91 %, 98.3 %.
- **Visibility is not better than a constant.** Always answering "visible" is right 96.2 % of the time for the
  beak and 96.0 % for "either eye" on the test split; the head gets 97.7 % and 95.4 %. Do not use it to decide
  whether a head is in view.
- By eye: 32 CUB test crops (a contact sheet the training run writes) and 48 real detector crops of the evaluation set: clear
  birds (herons, geese, falcons, hummingbirds, gulls) are pointed accurately (*notes*).

## Flying or not (`flight.py`, `flight.safetensors`, 16 KB)

**Shape.** Mean of the 256 block-15 tokens -> standardise (stored mean and std) -> Linear(1280, 1) -> sigmoid.

**Data.** iNaturalist observations whose observers filled in an observation field
(`fetch_flight.py`; one photo per observation, its first licensed one, 500 px; no observation or photo of the
evaluation set):

| Label | Observation field (id) | Photos (*notes*) |
|---|---|---|
| flying | "birds in flight" (16392), any value | 1,231 |
| flying | "Behavior: flying" (6631) = yes | 452 |
| flying | "Flying?" (3674) = yes | 1,200 |
| not flying | "Flying?" (3674) = no | 1,999 |

The per-field counts are from the notes and add up to 4,882; the download manifest has no field column and
lists 2,883 flying and 2,000 not flying. Listed 4,883 photos (licences: CC BY-NC 4,245, CC BY 528, CC0 110), 4,881 downloaded (449 MB, *notes*), 4,876
read and used: 2,878 flying, 1,998 not. The label is the observer's word about the observation, not about this
photo: expect some noise. "Display flight" (1829) is a courtship display and was not used.

**Training.** Logistic regression (L-BFGS, 300 iterations, L2 1e-2) on whole photos framed like the tower's
input (shortest side 224, centre crop); it is used on detector crops, which it was not trained on. The shipped
head is fit on all 4,876 photos; the scores below come from fits that held the test photos out.

```
./run ../bioclip-bird-heads/fetch_flight.py                  # -> data/flight-raw (network: about 35 API calls, 0.5 GB)
./run distill/embed.py --src data/flight-raw --out data/flight
./run ../bioclip-bird-heads/flight.py --out runs/flight-1
```

**Test.**

| Held out | n | correct | flying found | not flying kept |
|---|---|---|---|---|
| one species in five (by a hash of the name): birds it never saw | 743 | 97.4 % | 96.0 % | 99.4 % |
| photographers who posted both kinds | 1,011 | 97.5 % | 96.6 % | 98.3 % |

The second split is the check against fooling ourselves: there the head cannot score by recognising who took
the photo (camera, place, style). By eye (*notes*): of 400 crops of the evaluation set it called 92 flying; the
40 it was surest of were all flying, and 24 random "not flying" were all right.

## Known failures

From looking at predictions on the evaluation set and on 94 detector crops of a private shoot (*notes*):

- **Small, far birds** and **flocks**: the beak and eyes still get points, often wrong; with several birds in the
  crop the beak and the eye can land on different birds.
- **No bird in the box** (wire, fence, feathers, out-of-focus clutter): both heads still answer. The beak was
  called visible on 96 % of those 94 crops, and 4 of the 6 crops called flying were blurred or birdless boxes
  (P 0.55-0.79).
- **Spread wings at rest**: a turkey vulture sunning on a tower with open wings was called flying (0.78).
- Not done: a comparison with labels from SigLIP2 or OWLv2 prompts, training the flight head on crops, and any
  measurement of the eye points' use for sharpness.
