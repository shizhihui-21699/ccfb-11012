# Reference images

Four 600x600 crops of public-domain photographs, used as a *realistic*
baseline by `tests/test_real_photos.py`.

They come from Wikimedia Commons files marked "Public domain" (US
Government / NARA collections and pre-1929 artworks); the crops are
redistributed here only as test fixtures.

Synthetic covers in `stegscan/corpus.py` are useful for parametrised
sweeps, but their histograms are smoother than photography's. These four
files keep the calibration and the regression tests anchored to real
image statistics.
