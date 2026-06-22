"""A small benchmark of seeded-bug problems for measuring agent effectiveness.

Each problem: a buggy `buggy.py` + a deterministic `test_buggy.py` verifier
(exit 0 = fixed). Bugs are real but unambiguous, and every verifier terminates
fast (no infinite loops). pass@1 over these is our effectiveness metric pre-Step-9.
"""

PROBLEMS = [
    {
        "name": "merge_intervals",
        "buggy": '''\
def merge_intervals(intervals):
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] < last[1]:        # BUG: should be <=
            last[1] = current[1]         # BUG: should be max(last[1], current[1])
        else:
            merged.append(current)
    return merged
''',
        "test": '''\
from buggy import merge_intervals
assert merge_intervals([[1,4],[4,5]]) == [[1,5]], merge_intervals([[1,4],[4,5]])
assert merge_intervals([[1,6],[2,3]]) == [[1,6]], merge_intervals([[1,6],[2,3]])
assert merge_intervals([[1,2],[3,4]]) == [[1,2],[3,4]]
print("ok")
''',
    },
    {
        "name": "binary_search",
        "buggy": '''\
def binary_search(arr, target):
    lo, hi = 0, len(arr) - 1
    while lo < hi:                  # BUG: should be lo <= hi
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
''',
        "test": '''\
from buggy import binary_search
assert binary_search([1,3,5,7,9], 9) == 4
assert binary_search([1,3,5,7,9], 1) == 0
assert binary_search([1,3,5,7,9], 4) == -1
print("ok")
''',
    },
    {
        "name": "factorial",
        "buggy": '''\
def factorial(n):
    if n == 0:
        return 0          # BUG: base case should be 1
    return n * factorial(n - 1)
''',
        "test": '''\
from buggy import factorial
assert factorial(0) == 1
assert factorial(5) == 120
assert factorial(1) == 1
print("ok")
''',
    },
    {
        "name": "is_palindrome",
        "buggy": '''\
def is_palindrome(s):
    return s == s[::1]    # BUG: [::1] never reverses; should be [::-1]
''',
        "test": '''\
from buggy import is_palindrome
assert is_palindrome("racecar") is True
assert is_palindrome("hello") is False
assert is_palindrome("a") is True
print("ok")
''',
    },
    {
        "name": "fizzbuzz",
        "buggy": '''\
def fizzbuzz(n):
    out = []
    for i in range(1, n + 1):
        if i % 3 == 0:               # BUG: %15 must be checked FIRST
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        elif i % 15 == 0:
            out.append("FizzBuzz")
        else:
            out.append(str(i))
    return out
''',
        "test": '''\
from buggy import fizzbuzz
r = fizzbuzz(15)
assert r[14] == "FizzBuzz", r[14]
assert r[2] == "Fizz" and r[4] == "Buzz"
assert r[0] == "1"
print("ok")
''',
    },
]
