"""Exam suite: Lua, TypeScript and JavaScript tasks that are NEVER used for training.

The core benchmark in tasks.py mostly measures task families the model was fine-tuned on.
This exam measures something else: can the model solve new, slightly harder function-writing
tasks in each target language? Every family here is absent from tasks.py and from the SFT data
(see scripts/import_code_sft.py, which drops rows that overlap these prompts).

Expected outputs come from the Python reference implementation, so tests are correct by
construction. Each family also carries a reference solution per language, and the test suite
checks that those pass, which proves every task is solvable in that language as stated
(Lua tables, 1-based loops, integer division, JSON comparison).

Values are limited to integers, strings, booleans and lists of them, so they mean the same thing
in all three languages. No task returns an index, because indices differ between Lua and JS.
"""
import json
import random
import re

LANGUAGES = {"lua": "Lua", "typescript": "TypeScript", "javascript": "JavaScript"}
WORDS = ("amber basil cedar delta ember fable grove harbor island jungle kettle lantern meadow "
         "nickel orbit pepper quartz ribbon saddle timber umber velvet willow yonder zephyr "
         "apple echo idle oven under").split()

F = []


def family(name, params, returns, desc, gen, ref, lua, js, edge=()):
    """params: [(name, typescript_type)]. desc uses the parameter names literally."""
    F.append(dict(name=name, params=params, returns=returns, desc=desc, gen=gen, ref=ref,
                  lua=lua, js=js, edge=list(edge)))


def ints(r, lo=-9, hi=9, n=(0, 8)):
    return [r.randint(lo, hi) for _ in range(r.randint(*n))]


def text(r, n=(0, 5), caps=0.0):
    words = [r.choice(WORDS) for _ in range(r.randint(*n))]
    return " ".join(w.capitalize() if r.random() < caps else w for w in words)


def grid(r):
    rows, cols = r.randint(1, 4), r.randint(1, 4)
    return [[r.randint(-5, 9) for _ in range(cols)] for _ in range(rows)]


def _bucket_counts(nums, width):
    if not nums:
        return []
    out = [0] * (max(nums) // width + 1)
    for x in nums:
        out[x // width] += 1
    return out


def _nearest(n, m):
    low = n // m * m
    return low if n - low < low + m - n else low + m


def _leaders(nums):
    out, best = [], None
    for x in reversed(nums):
        if best is None or x > best:
            out.insert(0, x)
            best = x
    return out


def _runs(items):
    out = []
    for i, x in enumerate(items):
        if i and x == items[i - 1]:
            out[-1] += 1
        else:
            out.append(1)
    return out


def _longest_run(nums):
    best = run = 0
    for i, x in enumerate(nums):
        run = run + 1 if i and x > nums[i - 1] else 1
        best = max(best, run)
    return best


def _spaced(r):
    words = [r.choice(WORDS) for _ in range(r.randint(0, 4))]
    gap = lambda: " " * r.randint(1, 3)
    body = "".join(w + gap() for w in words).rstrip(" ") if words else ""
    return " " * r.randint(0, 2) + body + " " * r.randint(0, 2)


def _camel(r):
    words = [r.choice(WORDS) for _ in range(r.randint(1, 4))]
    return words[0] + "".join(w.capitalize() for w in words[1:])


def _mixed_text(r):
    parts = [r.choice(WORDS + ["Hello", "World", "42", "x9", "ok!"]) for _ in range(r.randint(0, 4))]
    return " ".join(parts)


# ---- lists of integers -----------------------------------------------------------------------
family("count_peaks", [("nums", "number[]")], "number",
       "returns how many elements of nums are strictly greater than both of their neighbours "
       "(the first and the last element never count)",
       lambda r: [ints(r, 0, 9, (0, 10))],
       lambda a: sum(1 for i in range(1, len(a) - 1) if a[i] > a[i - 1] and a[i] > a[i + 1]),
       "local function count_peaks(nums)\n  local count = 0\n  for i = 2, #nums - 1 do\n"
       "    if nums[i] > nums[i - 1] and nums[i] > nums[i + 1] then count = count + 1 end\n"
       "  end\n  return count\nend\n",
       "function count_peaks(nums) {\n  let count = 0;\n  for (let i = 1; i < nums.length - 1; i++) {\n"
       "    if (nums[i] > nums[i - 1] && nums[i] > nums[i + 1]) count++;\n  }\n  return count;\n}\n",
       edge=[[[]], [[5]], [[1, 3, 3, 1]]])
family("longest_increasing_run", [("nums", "number[]")], "number",
       "returns the length of the longest block of consecutive elements of nums in which every element "
       "is strictly greater than the one before it (0 for an empty list)",
       lambda r: [ints(r, 0, 9, (0, 10))], _longest_run,
       "local function longest_increasing_run(nums)\n  if #nums == 0 then return 0 end\n"
       "  local best, run = 1, 1\n  for i = 2, #nums do\n"
       "    if nums[i] > nums[i - 1] then run = run + 1 else run = 1 end\n"
       "    if run > best then best = run end\n  end\n  return best\nend\n",
       "function longest_increasing_run(nums) {\n  if (nums.length === 0) return 0;\n  let best = 1, run = 1;\n"
       "  for (let i = 1; i < nums.length; i++) {\n    run = nums[i] > nums[i - 1] ? run + 1 : 1;\n"
       "    best = Math.max(best, run);\n  }\n  return best;\n}\n",
       edge=[[[]], [[4, 4, 4]]])
family("rotate_right", [("nums", "number[]"), ("k", "number")], "number[]",
       "returns a new list with the elements of nums rotated k places to the right; k is never negative "
       "and may be larger than the length",
       lambda r: [ints(r, 0, 20, (0, 7)), r.randint(0, 12)],
       lambda a, k: a[len(a) - k % len(a):] + a[:len(a) - k % len(a)] if a else [],
       "local function rotate_right(nums, k)\n  local n, out = #nums, {}\n  if n == 0 then return out end\n"
       "  for i = 1, n do out[(i + k - 1) % n + 1] = nums[i] end\n  return out\nend\n",
       "function rotate_right(nums, k) {\n  const n = nums.length;\n  if (n === 0) return [];\n"
       "  const shift = k % n;\n  return nums.slice(n - shift).concat(nums.slice(0, n - shift));\n}\n",
       edge=[[[], 3], [[1, 2, 3], 0], [[1, 2, 3], 3]])
family("chunk_sums", [("nums", "number[]"), ("size", "number")], "number[]",
       "splits nums into consecutive groups of size elements (the last group may be shorter) and returns "
       "the sum of each group",
       lambda r: [ints(r, -9, 9, (0, 9)), r.randint(1, 4)],
       lambda a, s: [sum(a[i:i + s]) for i in range(0, len(a), s)],
       "local function chunk_sums(nums, size)\n  local out = {}\n  for i = 1, #nums, size do\n"
       "    local s = 0\n    for j = i, math.min(i + size - 1, #nums) do s = s + nums[j] end\n"
       "    out[#out + 1] = s\n  end\n  return out\nend\n",
       "function chunk_sums(nums, size) {\n  const out = [];\n  for (let i = 0; i < nums.length; i += size) {\n"
       "    out.push(nums.slice(i, i + size).reduce((s, x) => s + x, 0));\n  }\n  return out;\n}\n",
       edge=[[[], 2]])
family("leaders", [("nums", "number[]")], "number[]",
       "returns, in their original order, the elements of nums that are strictly greater than every "
       "element after them (the last element always qualifies)",
       lambda r: [ints(r, 0, 9, (0, 9))], _leaders,
       "local function leaders(nums)\n  local out, best = {}, nil\n  for i = #nums, 1, -1 do\n"
       "    if best == nil or nums[i] > best then\n      table.insert(out, 1, nums[i])\n      best = nums[i]\n"
       "    end\n  end\n  return out\nend\n",
       "function leaders(nums) {\n  const out = [];\n  let best = -Infinity;\n"
       "  for (let i = nums.length - 1; i >= 0; i--) {\n    if (nums[i] > best) {\n      out.unshift(nums[i]);\n"
       "      best = nums[i];\n    }\n  }\n  return out;\n}\n",
       edge=[[[]], [[7, 7, 7]]])
family("total_climb", [("heights", "number[]")], "number",
       "walks through heights from left to right and returns the total of all upward steps "
       "(each rise from one height to a higher next height adds the difference; drops add nothing)",
       lambda r: [ints(r, 0, 15, (0, 9))],
       lambda h: sum(max(0, h[i] - h[i - 1]) for i in range(1, len(h))),
       "local function total_climb(heights)\n  local total = 0\n  for i = 2, #heights do\n"
       "    if heights[i] > heights[i - 1] then total = total + heights[i] - heights[i - 1] end\n"
       "  end\n  return total\nend\n",
       "function total_climb(heights) {\n  let total = 0;\n  for (let i = 1; i < heights.length; i++) {\n"
       "    if (heights[i] > heights[i - 1]) total += heights[i] - heights[i - 1];\n  }\n  return total;\n}\n",
       edge=[[[]], [[3]]])
family("evens_then_odds", [("nums", "number[]")], "number[]",
       "returns the even numbers of nums followed by the odd numbers, keeping the original order inside "
       "each group",
       lambda r: [ints(r)],
       lambda a: [x for x in a if x % 2 == 0] + [x for x in a if x % 2],
       "local function evens_then_odds(nums)\n  local out = {}\n"
       "  for _, x in ipairs(nums) do if x % 2 == 0 then out[#out + 1] = x end end\n"
       "  for _, x in ipairs(nums) do if x % 2 ~= 0 then out[#out + 1] = x end end\n  return out\nend\n",
       "function evens_then_odds(nums) {\n  return nums.filter(x => x % 2 === 0).concat(nums.filter(x => x % 2 !== 0));\n}\n",
       edge=[[[]]])
family("top_k_distinct", [("nums", "number[]"), ("k", "number")], "number[]",
       "returns the k largest distinct values of nums from largest to smallest (all distinct values if "
       "there are fewer than k)",
       lambda r: [ints(r, 0, 9, (0, 9)), r.randint(0, 5)],
       lambda a, k: sorted(set(a), reverse=True)[:k],
       "local function top_k_distinct(nums, k)\n  local seen, values = {}, {}\n"
       "  for _, x in ipairs(nums) do\n    if not seen[x] then\n      seen[x] = true\n"
       "      values[#values + 1] = x\n    end\n  end\n"
       "  table.sort(values, function(p, q) return p > q end)\n  local out = {}\n"
       "  for i = 1, math.min(k, #values) do out[i] = values[i] end\n  return out\nend\n",
       "function top_k_distinct(nums, k) {\n  return [...new Set(nums)].sort((p, q) => q - p).slice(0, k);\n}\n",
       edge=[[[], 2], [[3, 3, 3], 2]])
family("zip_sum", [("a", "number[]"), ("b", "number[]")], "number[]",
       "adds a and b position by position; when one list is longer, its extra elements are kept "
       "(a missing element counts as 0)",
       lambda r: [ints(r, -9, 9, (0, 5)), ints(r, -9, 9, (0, 5))],
       lambda a, b: [(a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(max(len(a), len(b)))],
       "local function zip_sum(a, b)\n  local out = {}\n"
       "  for i = 1, math.max(#a, #b) do out[i] = (a[i] or 0) + (b[i] or 0) end\n  return out\nend\n",
       "function zip_sum(a, b) {\n  const out = [];\n  for (let i = 0; i < Math.max(a.length, b.length); i++) {\n"
       "    out.push((a[i] ?? 0) + (b[i] ?? 0));\n  }\n  return out;\n}\n",
       edge=[[[], []], [[1, 2, 3], []]])
family("count_inversions", [("nums", "number[]")], "number",
       "returns how many pairs of positions in nums have a larger number appearing before a smaller one",
       lambda r: [ints(r, 0, 9, (0, 9))],
       lambda a: sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] > a[j]),
       "local function count_inversions(nums)\n  local count = 0\n  for i = 1, #nums do\n"
       "    for j = i + 1, #nums do\n      if nums[i] > nums[j] then count = count + 1 end\n    end\n"
       "  end\n  return count\nend\n",
       "function count_inversions(nums) {\n  let count = 0;\n  for (let i = 0; i < nums.length; i++) {\n"
       "    for (let j = i + 1; j < nums.length; j++) {\n      if (nums[i] > nums[j]) count++;\n    }\n"
       "  }\n  return count;\n}\n",
       edge=[[[]], [[1, 2, 3]]])
family("alternating_sum", [("nums", "number[]")], "number",
       "returns the first element of nums minus the second plus the third minus the fourth, and so on "
       "(0 for an empty list)",
       lambda r: [ints(r)],
       lambda a: sum(x if i % 2 == 0 else -x for i, x in enumerate(a)),
       "local function alternating_sum(nums)\n  local total = 0\n  for i, x in ipairs(nums) do\n"
       "    if i % 2 == 1 then total = total + x else total = total - x end\n  end\n  return total\nend\n",
       "function alternating_sum(nums) {\n  return nums.reduce((t, x, i) => (i % 2 === 0 ? t + x : t - x), 0);\n}\n",
       edge=[[[]]])
family("count_pair_sums", [("nums", "number[]"), ("target", "number")], "number",
       "returns how many pairs of different positions in nums hold two numbers that add up to target "
       "(each pair of positions is counted once)",
       lambda r: [ints(r, 0, 9, (0, 8)), r.randint(0, 18)],
       lambda a, t: sum(1 for i in range(len(a)) for j in range(i + 1, len(a)) if a[i] + a[j] == t),
       "local function count_pair_sums(nums, target)\n  local count = 0\n  for i = 1, #nums do\n"
       "    for j = i + 1, #nums do\n      if nums[i] + nums[j] == target then count = count + 1 end\n"
       "    end\n  end\n  return count\nend\n",
       "function count_pair_sums(nums, target) {\n  let count = 0;\n  for (let i = 0; i < nums.length; i++) {\n"
       "    for (let j = i + 1; j < nums.length; j++) {\n      if (nums[i] + nums[j] === target) count++;\n"
       "    }\n  }\n  return count;\n}\n",
       edge=[[[], 4], [[2, 2, 2], 4]])
family("drop_every_kth", [("nums", "number[]"), ("k", "number")], "number[]",
       "returns nums without the k-th, 2k-th, 3k-th, ... elements, counting positions from 1 "
       "(k is at least 1)",
       lambda r: [ints(r, 0, 30, (0, 9)), r.randint(1, 4)],
       lambda a, k: [x for i, x in enumerate(a, 1) if i % k],
       "local function drop_every_kth(nums, k)\n  local out = {}\n  for i, x in ipairs(nums) do\n"
       "    if i % k ~= 0 then out[#out + 1] = x end\n  end\n  return out\nend\n",
       "function drop_every_kth(nums, k) {\n  return nums.filter((_, i) => (i + 1) % k !== 0);\n}\n",
       edge=[[[], 2], [[1, 2, 3], 1]])
family("bucket_counts", [("nums", "number[]"), ("width", "number")], "number[]",
       "counts the non-negative numbers of nums in buckets of the given width: bucket 0 holds values "
       "from 0 to width-1, bucket 1 holds width to 2*width-1, and so on; return the counts from bucket 0 "
       "up to the bucket of the largest value (an empty list gives an empty result)",
       lambda r: [ints(r, 0, 29, (0, 8)), r.randint(1, 10)], _bucket_counts,
       "local function bucket_counts(nums, width)\n  local out, top = {}, -1\n"
       "  for _, x in ipairs(nums) do if x // width > top then top = x // width end end\n"
       "  for i = 1, top + 1 do out[i] = 0 end\n"
       "  for _, x in ipairs(nums) do out[x // width + 1] = out[x // width + 1] + 1 end\n  return out\nend\n",
       "function bucket_counts(nums, width) {\n  if (nums.length === 0) return [];\n"
       "  const out = new Array(Math.floor(Math.max(...nums) / width) + 1).fill(0);\n"
       "  for (const x of nums) out[Math.floor(x / width)]++;\n  return out;\n}\n",
       edge=[[[], 5], [[0], 3]])
family("interleave", [("a", "number[]"), ("b", "number[]")], "number[]",
       "returns a list that takes elements from a and b in turn (a first); when one list runs out, the "
       "rest of the other list follows",
       lambda r: [ints(r, 0, 20, (0, 5)), ints(r, 0, 20, (0, 5))],
       lambda a, b: [x for i in range(max(len(a), len(b))) for x in (a[i:i + 1] + b[i:i + 1])],
       "local function interleave(a, b)\n  local out = {}\n  for i = 1, math.max(#a, #b) do\n"
       "    if i <= #a then out[#out + 1] = a[i] end\n    if i <= #b then out[#out + 1] = b[i] end\n"
       "  end\n  return out\nend\n",
       "function interleave(a, b) {\n  const out = [];\n  for (let i = 0; i < Math.max(a.length, b.length); i++) {\n"
       "    if (i < a.length) out.push(a[i]);\n    if (i < b.length) out.push(b[i]);\n  }\n  return out;\n}\n",
       edge=[[[], []], [[1, 2], []]])
# ---- grids -----------------------------------------------------------------------------------
family("column_sums", [("grid", "number[][]")], "number[]",
       "returns the sum of each column of grid, a non-empty list of rows that all have the same length",
       lambda r: [grid(r)], lambda g: [sum(row[c] for row in g) for c in range(len(g[0]))],
       "local function column_sums(grid)\n  local out = {}\n  for c = 1, #grid[1] do\n    local s = 0\n"
       "    for r = 1, #grid do s = s + grid[r][c] end\n    out[c] = s\n  end\n  return out\nend\n",
       "function column_sums(grid) {\n  return grid[0].map((_, c) => grid.reduce((s, row) => s + row[c], 0));\n}\n",
       edge=[[[[4]]]])
family("border_sum", [("grid", "number[][]")], "number",
       "returns the sum of the cells on the outer edge of grid (first and last row, first and last "
       "column), counting each cell once; grid is a non-empty list of equal-length rows",
       lambda r: [grid(r)],
       lambda g: sum(v for i, row in enumerate(g) for j, v in enumerate(row)
                     if i in (0, len(g) - 1) or j in (0, len(row) - 1)),
       "local function border_sum(grid)\n  local rows, cols, s = #grid, #grid[1], 0\n  for r = 1, rows do\n"
       "    for c = 1, cols do\n      if r == 1 or r == rows or c == 1 or c == cols then s = s + grid[r][c] end\n"
       "    end\n  end\n  return s\nend\n",
       "function border_sum(grid) {\n  const rows = grid.length, cols = grid[0].length;\n  let s = 0;\n"
       "  for (let r = 0; r < rows; r++) {\n    for (let c = 0; c < cols; c++) {\n"
       "      if (r === 0 || r === rows - 1 || c === 0 || c === cols - 1) s += grid[r][c];\n    }\n  }\n"
       "  return s;\n}\n",
       edge=[[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])
# ---- numbers ---------------------------------------------------------------------------------
family("count_divisors", [("n", "number")], "number",
       "returns how many positive whole numbers divide n exactly (n is at least 1)",
       lambda r: [r.randint(1, 400)], lambda n: sum(1 for d in range(1, n + 1) if n % d == 0),
       "local function count_divisors(n)\n  local count = 0\n"
       "  for d = 1, n do if n % d == 0 then count = count + 1 end end\n  return count\nend\n",
       "function count_divisors(n) {\n  let count = 0;\n  for (let d = 1; d <= n; d++) if (n % d === 0) count++;\n"
       "  return count;\n}\n",
       edge=[[1], [360]])
family("bit_count_range", [("a", "number"), ("b", "number")], "number",
       "returns the total number of 1 bits in the binary forms of all whole numbers from a to b "
       "inclusive (0 <= a <= b)",
       lambda r: (lambda a: [a, a + r.randint(0, 60)])(r.randint(0, 240)),
       lambda a, b: sum(bin(x).count("1") for x in range(a, b + 1)),
       "local function bit_count_range(a, b)\n  local total = 0\n  for x = a, b do\n    local v = x\n"
       "    while v > 0 do\n      total = total + v % 2\n      v = v // 2\n    end\n  end\n  return total\nend\n",
       "function bit_count_range(a, b) {\n  let total = 0;\n  for (let x = a; x <= b; x++) {\n"
       "    for (let v = x; v > 0; v = Math.floor(v / 2)) total += v % 2;\n  }\n  return total;\n}\n",
       edge=[[0, 0], [7, 7]])
family("nearest_multiple", [("n", "number"), ("m", "number")], "number",
       "returns the multiple of m that is closest to n; when two multiples are equally close, return "
       "the larger one (m is positive, n may be negative)",
       lambda r: [r.randint(-100, 100), r.randint(1, 12)], _nearest,
       "local function nearest_multiple(n, m)\n  local low = (n // m) * m\n  local high = low + m\n"
       "  if n - low < high - n then return low end\n  return high\nend\n",
       "function nearest_multiple(n, m) {\n  const low = Math.floor(n / m) * m, high = low + m;\n"
       "  return n - low < high - n ? low : high;\n}\n",
       edge=[[-5, 10], [15, 10], [-7, 3]])
family("zero_pad", [("nums", "number[]"), ("width", "number")], "string[]",
       "returns each non-negative number of nums as a string, padded on the left with zeros to at least "
       "width characters (longer numbers stay as they are)",
       lambda r: [[r.choice([r.randint(0, 9), r.randint(10, 999), r.randint(1000, 99999)])
                   for _ in range(r.randint(0, 5))], r.randint(1, 5)],
       lambda a, w: [str(x).zfill(w) for x in a],
       "local function zero_pad(nums, width)\n  local out = {}\n"
       "  for i, x in ipairs(nums) do out[i] = string.format(\"%0\" .. width .. \"d\", x) end\n  return out\nend\n",
       "function zero_pad(nums, width) {\n  return nums.map(x => String(x).padStart(width, \"0\"));\n}\n",
       edge=[[[], 3], [[0], 1]])
# ---- strings ---------------------------------------------------------------------------------
family("run_lengths", [("items", "string[]")], "number[]",
       "returns the lengths of the blocks of equal neighbouring strings in items, from left to right",
       lambda r: [[r.choice("abc") for _ in range(r.randint(0, 9))]], _runs,
       "local function run_lengths(items)\n  local out = {}\n  for i = 1, #items do\n"
       "    if i > 1 and items[i] == items[i - 1] then out[#out] = out[#out] + 1 else out[#out + 1] = 1 end\n"
       "  end\n  return out\nend\n",
       "function run_lengths(items) {\n  const out = [];\n  for (let i = 0; i < items.length; i++) {\n"
       "    if (i > 0 && items[i] === items[i - 1]) out[out.length - 1]++;\n    else out.push(1);\n  }\n"
       "  return out;\n}\n",
       edge=[[[]], [["a", "a", "a"]]])
family("compress_spaces", [("s", "string")], "string",
       "returns s with every run of spaces turned into a single space and no spaces at the start or end",
       lambda r: [_spaced(r)], lambda s: " ".join(s.split()),
       "local function compress_spaces(s)\n  local words = {}\n"
       "  for w in s:gmatch(\"%S+\") do words[#words + 1] = w end\n  return table.concat(words, \" \")\nend\n",
       "function compress_spaces(s) {\n  return s.split(\" \").filter(w => w !== \"\").join(\" \");\n}\n",
       edge=[[""], ["   "]])
family("caesar_shift", [("s", "string"), ("k", "number")], "string",
       "moves every lowercase letter of s k places forward in the alphabet, wrapping from z back to a "
       "(k may be negative or larger than 26); all other characters stay the same",
       lambda r: [_mixed_text(r), r.randint(-30, 30)],
       lambda s, k: "".join(chr((ord(c) - 97 + k) % 26 + 97) if "a" <= c <= "z" else c for c in s),
       "local function caesar_shift(s, k)\n"
       "  return (s:gsub(\"%l\", function(c) return string.char((c:byte() - 97 + k) % 26 + 97) end))\nend\n",
       "function caesar_shift(s, k) {\n"
       "  return s.replace(/[a-z]/g, c => String.fromCharCode((((c.charCodeAt(0) - 97 + k) % 26) + 26) % 26 + 97));\n}\n",
       edge=[["", 3], ["xyz", 3]])
family("camel_to_snake", [("s", "string")], "string",
       "converts the camelCase name s (letters only, starting with a lowercase letter) to snake_case: "
       "every uppercase letter becomes an underscore followed by the same letter in lowercase",
       lambda r: [_camel(r)], lambda s: re.sub(r"[A-Z]", lambda m: "_" + m.group(0).lower(), s),
       "local function camel_to_snake(s)\n  return (s:gsub(\"%u\", function(c) return \"_\" .. c:lower() end))\nend\n",
       "function camel_to_snake(s) {\n  return s.replace(/[A-Z]/g, c => \"_\" + c.toLowerCase());\n}\n",
       edge=[["x"]])
family("digit_runs", [("s", "string")], "number[]",
       "finds every maximal run of digits in s and returns them as numbers in order (leading zeros are "
       "dropped, so \"007\" becomes 7)",
       lambda r: ["".join(r.choice("ab0123xyz9") for _ in range(r.randint(0, 12)))],
       lambda s: [int(d) for d in re.findall(r"[0-9]+", s)],
       "local function digit_runs(s)\n  local out = {}\n"
       "  for d in s:gmatch(\"%d+\") do out[#out + 1] = tonumber(d) end\n  return out\nend\n",
       "function digit_runs(s) {\n  return (s.match(/[0-9]+/g) || []).map(Number);\n}\n",
       edge=[[""], ["abc"], ["a007b"]])
family("mask_middle", [("s", "string")], "string",
       "keeps the first and last characters of s and replaces every character between them with *; "
       "strings of length 2 or less are returned unchanged",
       lambda r: [r.choice(WORDS)[:r.randint(1, 8)]],
       lambda s: s if len(s) <= 2 else s[0] + "*" * (len(s) - 2) + s[-1],
       "local function mask_middle(s)\n  if #s <= 2 then return s end\n"
       "  return s:sub(1, 1) .. string.rep(\"*\", #s - 2) .. s:sub(-1)\nend\n",
       "function mask_middle(s) {\n  if (s.length <= 2) return s;\n"
       "  return s[0] + \"*\".repeat(s.length - 2) + s[s.length - 1];\n}\n",
       edge=[[""], ["ab"], ["abc"]])
family("word_score", [("w", "string")], "number",
       "returns the score of the lowercase word w, where a is worth 1, b is worth 2, and so on up to z "
       "worth 26 (an empty word scores 0)",
       lambda r: [r.choice(WORDS)], lambda w: sum(ord(c) - 96 for c in w),
       "local function word_score(w)\n  local total = 0\n  for i = 1, #w do total = total + w:byte(i) - 96 end\n"
       "  return total\nend\n",
       "function word_score(w) {\n  let total = 0;\n  for (const c of w) total += c.charCodeAt(0) - 96;\n"
       "  return total;\n}\n",
       edge=[[""], ["z"]])
family("reverse_each_word", [("s", "string")], "string",
       "reverses the letters of every word in s but keeps the words in the same order; words are "
       "separated by single spaces",
       lambda r: [text(r, (0, 5), 0.3)], lambda s: " ".join(w[::-1] for w in s.split(" ")),
       "local function reverse_each_word(s)\n  return (s:gsub(\"%S+\", function(w) return w:reverse() end))\nend\n",
       "function reverse_each_word(s) {\n  return s.split(\" \").map(w => [...w].reverse().join(\"\")).join(\" \");\n}\n",
       edge=[[""]])
family("vowel_start_words", [("s", "string")], "number",
       "returns how many words of s begin with a vowel (a, e, i, o or u, upper or lower case); words are "
       "separated by spaces",
       lambda r: [text(r, (0, 6), 0.3)],
       lambda s: sum(1 for w in s.split() if w[0].lower() in "aeiou"),
       "local function vowel_start_words(s)\n  local count = 0\n  for w in s:gmatch(\"%S+\") do\n"
       "    if w:sub(1, 1):lower():match(\"[aeiou]\") then count = count + 1 end\n  end\n  return count\nend\n",
       "function vowel_start_words(s) {\n"
       "  return s.split(\" \").filter(w => w !== \"\" && \"aeiou\".includes(w[0].toLowerCase())).length;\n}\n",
       edge=[[""], ["Echo idle oven"]])

FAMILIES = {f["name"]: f for f in F}


def literal(value, language):
    """Write a test value the way it looks in the task's language."""
    if language == "lua":
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return "{" + ", ".join(literal(v, language) for v in value) + "}"
        return json.dumps(value)
    return json.dumps(value)


def ts_signature(fam):
    params = ", ".join(f"{n}: {t}" for n, t in fam["params"])
    return f"{fam['name']}({params}): {fam['returns']}"


def solution(fam, language):
    if language == "lua":
        return fam["lua"]
    if language == "typescript":  # same body as the JS reference, with typed signature
        first, rest = fam["js"].split("\n", 1)
        return f"function {ts_signature(fam)} {{\n{rest}"
    return fam["js"]


def make_task(fam, language, rng, cases=8):
    ref = fam["ref"]
    tests, seen = [], set()
    candidates = fam["edge"] + [fam["gen"](rng) for _ in range(cases * 4)]
    for args in candidates:
        key = json.dumps(args)
        if key in seen:
            continue
        seen.add(key)
        tests.append({"args": args, "expected": ref(*json.loads(key))})
        if len(tests) == cases:
            break
    rng.shuffle(tests)
    plain = lambda args: all(a not in ([], "") for a in args)
    spare = next((a for a in candidates if json.dumps(a) not in seen and plain(a)), None)
    if spare is not None:  # the example shown must not be a scored test
        example = {"args": spare, "expected": ref(*json.loads(json.dumps(spare)))}
    else:
        example = next((t for t in tests if plain(t["args"])), tests[0])
        tests.remove(example)
    name = fam["name"]
    if language == "typescript":
        signature = ts_signature(fam)
    else:
        signature = f"{name}({', '.join(n for n, _ in fam['params'])})"
    note = " Lists are Lua array tables." if language == "lua" else ""
    shown = ", ".join(literal(a, language) for a in example["args"])
    prompt = (f"Write a {LANGUAGES[language]} function `{signature}` that {fam['desc']}.{note}\n"
              f"Example: {name}({shown}) returns {literal(example['expected'], language)}")
    return {"family": name, "language": language, "function": name, "prompt": prompt,
            "tests": tests, "solutions": [solution(fam, language)], "split": "exam"}


def exam(seed=2027, per_family=1, languages=tuple(LANGUAGES)):
    """Fixed exam: every family in every language, `per_family` instances each."""
    rng = random.Random(seed)
    return [make_task(fam, language, rng) for fam in F for language in languages for _ in range(per_family)]


def prompts():
    """All exam prompt texts, for decontaminating training data."""
    return [t["prompt"] for t in exam()]
