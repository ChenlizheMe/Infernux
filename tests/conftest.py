"""Keep executable acceptance drivers separate from pytest test modules."""

collect_ignore = ["acceptance", "android", "fixtures", "gpu", "native", "release"]
