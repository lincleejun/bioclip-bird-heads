# bioclip-bird-heads

Two small add-on heads for [bioclip-bird](https://github.com/lincleejun/bioclip-bird)'s int8 BioCLIP 2.5 image
tower: **beak and eye points** and **flying or not**. They read the patch tokens the tower already computes
while it names the bird, so they add almost nothing to the cost of identification.

| Head | File | Size | Trained on | Held-out result |
|---|---|---|---|---|
| Beak + eyes | `parts.safetensors` | 1.3 MB | CUB-200-2011 train split (5,994 images) | CUB test: 97.8 % of beaks and 98.2 % of eyes within 5 % of the crop's side |
| Flying | `flight.safetensors` | 16 KB | 4,876 iNaturalist photos labelled by their observers | 97.4 % right on species it never saw |

Structure, data, training, tests and known failures: [docs/heads.md](docs/heads.md).

**The weights are for non-commercial use only** (see [WEIGHTS-LICENSE.md](WEIGHTS-LICENSE.md)). They live in a separate repository
from bioclip-bird so that its MIT-licensed files stay free of these restrictions.

## Download

Weights are attached to this repository's
[GitHub Releases](https://github.com/lincleejun/bioclip-bird-heads/releases), with `SHA256SUMS`. You also need,
from [bioclip-bird's releases](https://github.com/lincleejun/bioclip-bird/releases):
`bioclip-2.5-vith14-visual-int8.safetensors` (the tower) and `bioclip-2.5-vith14-avilist-2026-names.npz`
(the species names).

## Use

bioclip-bird's `examples/identify.py` runs the tower and, when given these files, both heads:

```
python examples/identify.py crop.jpg \
    --model bioclip-2.5-vith14-visual-int8.safetensors --names bioclip-2.5-vith14-avilist-2026-names.npz \
    --parts parts.safetensors --flight flight.safetensors
```

It prints the top five species, the beak and eye points in the crop's pixels with P(visible), and P(flying).
The input is a crop around one bird. The heads were trained on block 15 of that exact tower: they do not work
with another BioCLIP checkpoint or another block.

## Train

The three scripts here depend on bioclip-bird's `distill` package (`distill.slim` for the tower,
`distill.fetch_inat` for the download helpers) and on its development environment (gardener's, see
bioclip-bird's README): `parts.py` frames CUB birds with gardener's own crop rule. Clone this repository next
to bioclip-bird as `bioclip-bird-heads` and run from bioclip-bird:

```
./run ../bioclip-bird-heads/parts.py train --layer 15 --out runs/parts-1     # needs data/cub/CUB_200_2011
./run ../bioclip-bird-heads/fetch_flight.py                                  # network: about 0.5 GB
./run distill/embed.py --src data/flight-raw --out data/flight
./run ../bioclip-bird-heads/flight.py --out runs/flight-1
```

| File | Does |
|---|---|
| `parts.py` | Beak/eye head: CUB framing, block probe, training, test report |
| `flight.py` | Flight head: features, held-out tests, the shipped fit |
| `fetch_flight.py` | iNaturalist photos whose observers said whether the bird was flying |

## License

- **Code** (the `.py` files and docs): MIT, see [LICENSE](LICENSE).
- **Weights: non-commercial use only**, see [WEIGHTS-LICENSE.md](WEIGHTS-LICENSE.md) (also attached to each release).
