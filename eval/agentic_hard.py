"""HARD agentic bug-fix set — the real couple benchmark.

Each task is a small module with MULTIPLE bugs across functions, plus edge cases (empty input,
wraparound, normalization, off-by-one). The couple must read the file, reason about every bug,
rewrite it, and make a strict test suite pass end-to-end (read -> reason -> edit -> verify ->
retry). This is harder than heldout_multi (toy one-liners) and is where the couple, not just the
raw brain, has to carry it. Disjoint from SEEDS/training.
"""

PROBLEMS = [
    {"name": "roman_numerals",
     "buggy": (
         "def to_roman(n):\n"
         "    vals = [(1000,'M'),(500,'D'),(100,'C'),(50,'L'),(10,'X'),(5,'V'),(1,'I')]\n"
         "    out = ''\n"
         "    for v, sym in vals:\n"
         "        while n >= v:\n"
         "            out += sym\n"
         "            n -= v\n"
         "    return out\n"),   # bug: no subtractive forms (4,9,40,90,400,900)
     "test": (
         "from buggy import to_roman\n"
         "assert to_roman(4) == 'IV'\nassert to_roman(9) == 'IX'\n"
         "assert to_roman(40) == 'XL'\nassert to_roman(90) == 'XC'\n"
         "assert to_roman(2024) == 'MMXXIV'\nassert to_roman(3888) == 'MMMDCCCLXXXVIII'\n"
         "print('ok')\n")},
    {"name": "caesar_cipher",
     "buggy": (
         "def encrypt(s, k):\n"
         "    return ''.join(chr(ord(c) + k) for c in s)\n\n"   # bug: no wraparound, no case guard
         "def decrypt(s, k):\n"
         "    return encrypt(s, k)\n"),                          # bug: should be -k
     "test": (
         "from buggy import encrypt, decrypt\n"
         "assert encrypt('abc', 1) == 'bcd'\n"
         "assert encrypt('xyz', 3) == 'abc'\n"
         "assert encrypt('Hello, World!', 5) == 'Mjqqt, Btwqi!'\n"
         "assert decrypt(encrypt('Secret', 7), 7) == 'Secret'\n"
         "print('ok')\n")},
    {"name": "binary_search",
     "buggy": (
         "def bsearch(xs, target):\n"
         "    lo, hi = 0, len(xs)\n"           # bug: hi should be len-1 (or handle half-open)
         "    while lo <= hi:\n"               # bug: with hi=len this indexes out of range
         "        mid = (lo + hi) // 2\n"
         "        if xs[mid] == target:\n"
         "            return mid\n"
         "        if xs[mid] < target:\n"
         "            lo = mid\n"              # bug: infinite loop, should be mid+1
         "        else:\n"
         "            hi = mid - 1\n"
         "    return -1\n"),
     "test": (
         "from buggy import bsearch\n"
         "assert bsearch([1,3,5,7,9], 7) == 3\n"
         "assert bsearch([1,3,5,7,9], 1) == 0\n"
         "assert bsearch([1,3,5,7,9], 9) == 4\n"
         "assert bsearch([1,3,5,7,9], 4) == -1\n"
         "assert bsearch([], 1) == -1\nprint('ok')\n")},
    {"name": "word_frequency",
     "buggy": (
         "def word_count(text):\n"
         "    d = {}\n"
         "    for w in text.split():\n"        # bug: no lowercasing, no punctuation strip
         "        d[w] = d.get(w, 0) + 1\n"
         "    return d\n\n"
         "def top_word(text):\n"
         "    d = word_count(text)\n"
         "    return list(d.keys())[0]\n"),     # bug: not the max-count word
     "test": (
         "from buggy import word_count, top_word\n"
         "wc = word_count('The cat, the CAT! the dog.')\n"
         "assert wc['the'] == 3 and wc['cat'] == 2 and wc['dog'] == 1\n"
         "assert top_word('a a a b b c') == 'a'\nprint('ok')\n")},
    {"name": "matrix_ops",
     "buggy": (
         "def transpose(m):\n"
         "    return m\n\n"                     # bug: identity, not transpose
         "def row_sums(m):\n"
         "    return [sum(m[0]) for _ in m]\n"), # bug: always sums first row
     "test": (
         "from buggy import transpose, row_sums\n"
         "assert transpose([[1,2,3],[4,5,6]]) == [[1,4],[2,5],[3,6]]\n"
         "assert row_sums([[1,2],[3,4],[5,6]]) == [3,7,11]\n"
         "assert transpose([]) == []\nprint('ok')\n")},
    {"name": "fizzbuzz_range",
     "buggy": (
         "def fizzbuzz(n):\n"
         "    out = []\n"
         "    for i in range(n):\n"             # bug: should be 1..n inclusive
         "        if i % 3 == 0:\n"             # bug: order means 15 -> 'Fizz' not 'FizzBuzz'
         "            out.append('Fizz')\n"
         "        elif i % 5 == 0:\n"
         "            out.append('Buzz')\n"
         "        else:\n"
         "            out.append(str(i))\n"
         "    return out\n"),
     "test": (
         "from buggy import fizzbuzz\n"
         "r = fizzbuzz(15)\n"
         "assert len(r) == 15 and r[0] == '1'\n"
         "assert r[2] == 'Fizz' and r[4] == 'Buzz' and r[14] == 'FizzBuzz'\n"
         "print('ok')\n")},
    {"name": "stack_balance",
     "buggy": (
         "def is_balanced(s):\n"
         "    stack = []\n"
         "    pairs = {')':'(', ']':'[', '}':'{'}\n"
         "    for c in s:\n"
         "        if c in '([{':\n"
         "            stack.append(c)\n"
         "        elif c in ')]}':\n"
         "            stack.pop()\n"            # bug: pop on empty + no match check
         "    return True\n"),                  # bug: ignores leftover open brackets
     "test": (
         "from buggy import is_balanced\n"
         "assert is_balanced('()[]{}') is True\n"
         "assert is_balanced('([{}])') is True\n"
         "assert is_balanced('(]') is False\n"
         "assert is_balanced('(()') is False\n"
         "assert is_balanced(')(') is False\nprint('ok')\n")},
    {"name": "fraction_add",
     "buggy": (
         "def add_fractions(a, b):\n"
         "    # a, b are (num, den) tuples; return reduced (num, den)\n"
         "    num = a[0]*b[1] + b[0]*a[1]\n"
         "    den = a[1] * b[1]\n"
         "    return (num, den)\n"),            # bug: not reduced to lowest terms
     "test": (
         "from buggy import add_fractions\n"
         "assert add_fractions((1,2),(1,2)) == (1,1)\n"
         "assert add_fractions((1,3),(1,6)) == (1,2)\n"
         "assert add_fractions((2,4),(1,4)) == (3,4)\nprint('ok')\n")},
]
