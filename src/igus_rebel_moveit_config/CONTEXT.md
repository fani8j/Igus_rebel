# Ubiquitous Language

## Snapshot-relative combined plan
A plan whose transit, hover, cutter endpoint, and orientation targets are derived from one selected carton snapshot rather than a previously calibrated carton pose; in combined mode the middle seam endpoints and orientation remain exactly snapshot-derived while side cuts may use bounded local orientation offsets.

## Side-cut candidate
One contiguous interval on a detected carton side edge, inset independently from the two carton corners on the configured grid, paired with a cutter orientation offset around the snapshot-derived pose.

## Feasible side cut
A side-cut candidate that participates in a collision-free, fully planned combined trajectory within the configured search time budget.

## Best bounded combined plan
The feasible combined trajectory with the greatest total left-plus-right side-cut length found before the configured search deadline. It is not claimed globally maximal when the deadline expires.
