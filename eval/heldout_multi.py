"""HARDER held-out set: each file has 2+ INDEPENDENT bugs across different functions.

This is the multi-step frontier — the agent must decompose, fix each bug, and make a
single test suite pass. Disjoint from all training data and from heldout.py.

SEEDS are analogous SINGLE-pattern fixes (a temperature conversion, an initials join,
etc.). They are NOT the answers to these compound problems — they're the kind of prior
work a real brain would already have. Loading them for the brain-ON condition tests
whether recalling related building blocks helps the 3B solve NEW compound tasks. That's
a FAIR lift test (analogy), not memorization (the exact compound answer is never stored).
"""

PROBLEMS = [
    {"name": "temp_and_initials",
     "buggy": (
         "def to_celsius(f):\n    return (f - 32) * 5 / 9 + 1\n\n"
         "def initials(name):\n    return name[0]\n"),
     "test": (
         "from buggy import to_celsius, initials\n"
         "assert to_celsius(32) == 0\nassert to_celsius(212) == 100\n"
         "assert initials('ada lovelace') == 'AL'\nprint('ok')\n")},
    {"name": "mean_and_median",
     "buggy": (
         "def mean(xs):\n    return sum(xs) // len(xs)\n\n"
         "def median(xs):\n    return max(xs)\n"),
     "test": (
         "from buggy import mean, median\n"
         "assert mean([1,2,3,4]) == 2.5\n"
         "assert median([3,1,2]) == 2\nassert median([1,2,3,4]) == 2.5\nprint('ok')\n")},
    {"name": "vowels_and_reverse",
     "buggy": (
         "def count_vowels(s):\n    return sum(1 for c in s if c in 'xyz')\n\n"
         "def reverse_words(s):\n    return s\n"),
     "test": (
         "from buggy import count_vowels, reverse_words\n"
         "assert count_vowels('hello') == 2\n"
         "assert reverse_words('a b c') == 'c b a'\nprint('ok')\n")},
    {"name": "dedupe_and_flatten",
     "buggy": (
         "def dedupe(xs):\n    return xs\n\n"
         "def flatten(xss):\n    return xss\n"),
     "test": (
         "from buggy import dedupe, flatten\n"
         "assert dedupe([1,1,2,3,3]) == [1,2,3]\n"
         "assert flatten([[1,2],[3],[4,5]]) == [1,2,3,4,5]\nprint('ok')\n")},
    {"name": "gcd_and_lcm",
     "buggy": (
         "def gcd(a, b):\n    return a\n\n"
         "def lcm(a, b):\n    return a * b\n"),
     "test": (
         "from buggy import gcd, lcm\n"
         "assert gcd(12, 8) == 4\nassert lcm(4, 6) == 12\nprint('ok')\n")},
    {"name": "title_and_clamp",
     "buggy": (
         "def titlecase(s):\n    return s.lower()\n\n"
         "def clamp(x, lo, hi):\n    return x\n"),
     "test": (
         "from buggy import titlecase, clamp\n"
         "assert titlecase('hello world') == 'Hello World'\n"
         "assert clamp(15, 0, 10) == 10\nassert clamp(-3, 0, 10) == 0\nprint('ok')\n")},
]

# Analogous building-block knowledge (single, correct, generic). Brain-ON seeds these.
SEEDS = [
    ("convert fahrenheit to celsius",
     "def to_celsius(f):\n    return (f - 32) * 5 / 9\n"),
    ("initials: first letter of every word, uppercased",
     "def initials(name):\n    return ''.join(w[0] for w in name.split()).upper()\n"),
    ("average / mean of a list of numbers",
     "def mean(xs):\n    return sum(xs) / len(xs)\n"),
    ("median of a list of numbers",
     "def median(xs):\n    s = sorted(xs)\n    n = len(s)\n    m = n // 2\n"
     "    return s[m] if n % 2 else (s[m-1] + s[m]) / 2\n"),
    ("count vowels in a string",
     "def count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')\n"),
    ("reverse the order of words in a string",
     "def reverse_words(s):\n    return ' '.join(s.split()[::-1])\n"),
    ("remove duplicates from a list preserving order",
     "def dedupe(xs):\n    seen = []\n    for x in xs:\n        if x not in seen:\n"
     "            seen.append(x)\n    return seen\n"),
    ("flatten a list of lists one level",
     "def flatten(xss):\n    return [x for xs in xss for x in xs]\n"),
    ("greatest common divisor of two integers",
     "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"),
    ("least common multiple of two integers",
     "def lcm(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n"),  # gcd helper idiom
    ("title-case every word in a string",
     "def titlecase(s):\n    return ' '.join(w.capitalize() for w in s.split())\n"),
    ("clamp a number to a range",
     "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n"),
]
