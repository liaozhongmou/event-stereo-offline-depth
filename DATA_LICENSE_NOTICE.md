# Dataset attribution and local input copy

The input archive preserves the exact research inputs used in this revision.
It is not a new release of either benchmark by their original authors.

- MVSEC: Alex Zihao Zhu, Dinesh Thakur, Tolga Ozaslan, Bernd Pfrommer,
  Vijay Kumar and Kostas Daniilidis. *The Multi Vehicle Stereo Event Camera
  Dataset: An Event Camera Dataset for3D Perception*. IEEE Robotics and Automation
  Letters3(3),2032–2039(2018), DOI10.1109/LRA.2018.2800793.
  Official source: https://daniilidis-group.github.io/mvsec/
  The supplied NPZ files are the author's previously converted inputs, not fresh
  official download files. Their hashes and conversion interpretation are recorded.
- DSEC: Mathias Gehrig, Willem Aarents, Daniel Gehrig and Davide Scaramuzza.
  *DSEC: A Stereo Event Camera Dataset for Driving Scenarios*. IEEE Robotics and
  Automation Letters(2021), DOI10.1109/LRA.2021.3068942.
  Official source: https://dsec.ifi.uzh.ch/
  Sequence: thun_00_a. Exact official download URLs and SHA-256 appear in
  JEI/datasets/dsec/thun_00_a/download_manifest.json inside the input archive.

Both official dataset sites state Creative Commons Attribution-ShareAlike4.0
International (CC BY-SA4.0), checked2026-10-21.
License: https://creativecommons.org/licenses/by-sa/4.0/
Preserve attribution and follow the original terms for dataset material and
adaptations. Changes in this project include MVSEC NPZ conversion predating this
revision, DSEC coordinate rectification, predictions, evaluation and new plots.
The original datasets' authors do not endorse this method or these results.
This notice does not assign a license to unrelated project code or manuscript text.
