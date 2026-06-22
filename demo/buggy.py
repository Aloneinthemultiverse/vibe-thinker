def merge_intervals(intervals):
    # Merges overlapping intervals. intervals is a list of [start, end].
    if not intervals:
        return []
    # Use a sorted copy to avoid mutating the original list and to merge touching intervals
    sorted_intervals = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_intervals[0][:]]  # copy the first interval
    for current in sorted_intervals[1:]:
        last = merged[-1]
        # Use <= to merge intervals that touch (e.g., [1,2] and [2,3])
        if current[0] <= last[1]:
            last[1] = current[1]
        else:
            merged.append(current[:])
    return merged
