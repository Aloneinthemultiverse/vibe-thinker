"""HELD-OUT benchmark — bug-fix problems the model has NEVER seen in training.

The 5 problems in problems.py leaked into training (curated + harvested came from
them), so eval on those measures memorization. These 12 are disjoint from problems.py,
train/gen_synth.py SYNTH, and the ChatGPT set — a clean generalization test of the same
task TYPE. Run: python -m eval.run_eval 3 --heldout
"""

PROBLEMS = [
    {"name": "count_negatives",
     "buggy": "def count_negatives(xs):\n    return sum(1 for x in xs if x > 0)\n",
     "test": "from buggy import count_negatives\nassert count_negatives([-1,2,-3,4])==2\nassert count_negatives([1,2,3])==0\nprint('ok')\n"},
    {"name": "sum_even",
     "buggy": "def sum_even(xs):\n    return sum(x for x in xs if x % 2 == 1)\n",
     "test": "from buggy import sum_even\nassert sum_even([1,2,3,4])==6\nassert sum_even([1,3,5])==0\nprint('ok')\n"},
    {"name": "is_sorted",
     "buggy": "def is_sorted(xs):\n    for i in range(len(xs)):\n        if xs[i] > xs[i+1]:\n            return False\n    return True\n",
     "test": "from buggy import is_sorted\nassert is_sorted([1,2,3]) is True\nassert is_sorted([1,3,2]) is False\nassert is_sorted([5]) is True\nprint('ok')\n"},
    {"name": "digit_sum",
     "buggy": "def digit_sum(n):\n    total = 0\n    while n > 0:\n        total += n % 10\n        n //= 100\n    return total\n",
     "test": "from buggy import digit_sum\nassert digit_sum(123)==6\nassert digit_sum(9)==9\nprint('ok')\n"},
    {"name": "is_prime",
     "buggy": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, n):\n        if n % i == 0:\n            return True\n    return False\n",
     "test": "from buggy import is_prime\nassert is_prime(7) is True\nassert is_prime(9) is False\nassert is_prime(2) is True\nprint('ok')\n"},
    {"name": "char_count",
     "buggy": "def char_count(s):\n    d = {}\n    for c in s:\n        d[c] = 1\n    return d\n",
     "test": "from buggy import char_count\nassert char_count('aab')=={'a':2,'b':1}\nprint('ok')\n"},
    {"name": "remove_zeros",
     "buggy": "def remove_zeros(xs):\n    return [x for x in xs if x is not None]\n",
     "test": "from buggy import remove_zeros\nassert remove_zeros([0,1,0,2])==[1,2]\nassert remove_zeros([3,0,0,4])==[3,4]\nprint('ok')\n"},
    {"name": "cumulative_sum",
     "buggy": "def cumulative_sum(xs):\n    out = []\n    total = 0\n    for x in xs:\n        out.append(total)\n        total += x\n    return out\n",
     "test": "from buggy import cumulative_sum\nassert cumulative_sum([1,2,3])==[1,3,6]\nprint('ok')\n"},
    {"name": "longest_word",
     "buggy": "def longest_word(s):\n    words = s.split()\n    best = words[0]\n    for w in words:\n        if len(w) < len(best):\n            best = w\n    return best\n",
     "test": "from buggy import longest_word\nassert longest_word('a bb ccc')=='ccc'\nassert longest_word('hi there')=='there'\nprint('ok')\n"},
    {"name": "swap_case",
     "buggy": "def swap_case(s):\n    return s.upper()\n",
     "test": "from buggy import swap_case\nassert swap_case('AbC')=='aBc'\nprint('ok')\n"},
    {"name": "mode",
     "buggy": "def mode(xs):\n    return max(xs)\n",
     "test": "from buggy import mode\nassert mode([1,2,2,3])==2\nassert mode([4,4,4,1])==4\nprint('ok')\n"},
    {"name": "abs_diff",
     "buggy": "def abs_diff(a, b):\n    return a - b\n",
     "test": "from buggy import abs_diff\nassert abs_diff(3,7)==4\nassert abs_diff(7,3)==4\nprint('ok')\n"},
]
