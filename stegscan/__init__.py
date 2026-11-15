"""stegscan - LSB / DCT steganalysis for PNG, BMP and JPEG images.

Public entry points:

    from stegscan.scan import scan_file, scan_directory
    from stegscan.report import build_report

The package implements four detectors:

* chi-square attack (Westfeld & Pfitzmann) with a smooth-model
  least-squares payload estimate,
* RS analysis (Fridrich, Goljan & Du),
* sample-pairs analysis (Dumitrescu, Wu & Memon),
* DCT coefficient-histogram analysis for JPEG (JSteg-style LSB
  replacement in the quantised coefficients).

All three spatial detectors estimate the *embedding rate* p, defined
as payload bits per host pixel (i.e. p = 1 means one embedded bit
per pixel, which for 8-bit channels is the full capacity of the LSB
plane).  A rate is mapped back to a payload size in bytes by the
scanner.
"""

__version__ = "0.1.0"
