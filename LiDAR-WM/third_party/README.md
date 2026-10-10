# Optional third-party dependencies

The following repositories are intentionally excluded from this release. Clone
them only if you need the corresponding feature, keep their licenses, and place
them exactly as shown below.

```bash
cd LiDAR-WM
mkdir -p third_party
git clone https://github.com/facebookresearch/sonata.git third_party/sonata
git -C third_party/sonata checkout 18c09ff8d713494f78a8213792262b910977a65d

git clone https://github.com/zyc00/Point-SAM.git third_party/Point-SAM
git -C third_party/Point-SAM checkout 25f4fd9675479f3c951e830771595855ff7ed466

git clone https://github.com/cansdev/learned-suppression-3d.git learned-suppression-3d
git -C learned-suppression-3d checkout 06273c07856d356aaf9fa5498a79153d4aef39f8
```

Sonata code is Apache-2.0 and its released weights have their own terms.
Point-SAM and learned-suppression-3d have their own MIT licenses. Obtain model
weights from their original authors and follow all upstream installation and
usage requirements. Do not add downloaded weights to Git.
