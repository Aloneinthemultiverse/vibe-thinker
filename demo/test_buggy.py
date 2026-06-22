"""Deterministic test. Exit code 0 = pass, non-zero = fail."""
from buggy import merge_intervals


def main():
    cases = [
        ([[1, 4], [4, 5]], [[1, 5]]),          # touching intervals must merge
        ([[1, 6], [2, 3]], [[1, 6]]),          # contained interval must not shrink the merge
        ([[1, 3], [2, 6], [8, 10]], [[1, 6], [8, 10]]),
        ([], []),
        ([[1, 2]], [[1, 2]]),
    ]
    for args, expected in cases:
        got = merge_intervals([list(x) for x in args])
        assert got == expected, f"merge_intervals({args}) -> {got}, expected {expected}"
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
