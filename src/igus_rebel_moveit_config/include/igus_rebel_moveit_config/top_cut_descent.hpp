// Early stop of the top-cut (middle_only) fraction descent of box_h_cut_executor (2026-10-24, search reduction only).
//
// plan_h_cut for the middle seam retries a failed plan with the cut END pulled back along the seam in 5 % steps (100,
// 95, ... down to top_cut_min_cut_fraction). Every step replans the whole MTC task. A failure in a stage BEFORE the cut
// stroke cannot be cured by a shorter cut when that stage's inputs do not depend on the cut length: the transit to the
// seam-start hover (and the short align move) start from the same state and aim at the same hover pose for every
// fraction (the stroke START, the blade direction along the seam and the clearances do not change when only seam_end
// moves), and the wrist gate (validateFreeSpaceTransitWrist) refuses a transit, not the stroke. Evidence and the
// narrowing: scan_results/fallback_speed_20261024/README.md.
//
// Applies only to pattern middle_only with the stroke START fixed (not middle_cut_reverse_travel, where the start moves
// with the fraction) and without half_seam. The entry/descend stage is NOT in the set: with middle_cut_sag_m > 0 the
// sag profile (built from the shortened seam) shifts the start depth by up to the sag, so it is not exactly
// fraction-independent. The cut stroke, the finish lift and everything after are fraction-dependent by nature.
//
// Narrowing (data: 4 of 165 descents with such a failure later succeeded, always at the MINIMUM fraction): after a
// fraction-independent failure the descent does not abort, it skips the intermediate fractions and tries the minimum
// fraction ONCE (the most favourable cut length); if that fails too, the descent ends as before.
//
// Pure helpers (std only) so the rule is unit-testable without MoveIt.
#pragma once

#include <cmath>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace top_cut_descent
{
/// Fraction step of the descent (same constant as box_h_cut_executor's shortening loop).
constexpr double kStep = 0.05;

struct FailedStage
{
  std::string name;
  int solutions = 0;   // "(s/f)" of MTC's explainFailure line
  int failures = 0;
  std::string comment;
};

/// The stage lines of the FIRST "Failing stage(s):" block of an MTC explainFailure text ("<name> (<s>/<f>): <comment>").
/// Stage names may contain parentheses; the LAST " (<digits>/<digits>)" of a line is the counter.
inline std::vector<FailedStage> parseFailingStages(const std::string & explanation)
{
  static const std::regex line_re(R"(^(.*) \(([0-9]+)/([0-9]+)\)(?:: ?(.*))?$)");
  std::vector<FailedStage> out;
  std::istringstream in(explanation);
  std::string line;
  bool in_block = false;
  while (std::getline(in, line))
  {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line == "Failing stage(s):")
    {
      if (in_block) break;   // second block: the first one is the answer
      in_block = true;
      continue;
    }
    if (!in_block) continue;
    std::smatch m;
    // first non-stage line ends the block (an interleaved log line starts with '[')
    if (line.empty() || line[0] == '[' || !std::regex_match(line, m, line_re)) break;
    FailedStage s;
    s.name = m[1].str();
    s.solutions = std::stoi(m[2].str());
    s.failures = std::stoi(m[3].str());
    s.comment = m[4].matched ? m[4].str() : std::string();
    out.push_back(s);
  }
  return out;
}

struct Classification
{
  bool before_cut_stroke = false;   // fraction-independent failure (see the header comment)
  std::string stage;                // the failing stage (or transit)
  std::string reason;               // short text for the log line
};

/// MTC failure: before the cut stroke iff at least one stage is listed and EVERY listed failing stage has no solution
/// and is one of `pre_stroke_stages` (names of the stages the planner built before the cut stroke whose inputs are
/// identical for every fraction). Anything else (empty list, a stroke/lift stage, an unknown stage) is not.
inline Classification classifyMtcFailure(const std::string & explanation, const std::set<std::string> & pre_stroke_stages)
{
  Classification c;
  const auto stages = parseFailingStages(explanation);
  if (stages.empty()) return c;
  c.stage = stages.front().name;
  c.reason = stages.front().comment;
  for (const auto & s : stages)
    if (s.solutions != 0 || pre_stroke_stages.count(s.name) == 0U) return c;
  c.before_cut_stroke = true;
  return c;
}

/// Wrist gate refusal: "planned free-space transit winds the wrist: transit '<stage>' joint6 travel X rad exceeds Y rad"
/// for a transit in `pre_stroke_transits`. Every other text of the gate (bookkeeping, "refusing to guess") is not
/// classified (the descent continues as before).
inline Classification classifyWristGateFailure(const std::string & error, const std::set<std::string> & pre_stroke_transits)
{
  Classification c;
  static const std::regex gate_re(R"(^planned free-space transit winds the wrist: transit '(.*)' joint6 travel .* rad exceeds .* rad$)");
  std::smatch m;
  if (!std::regex_match(error, m, gate_re)) return c;
  c.stage = m[1].str();
  c.reason = error;
  c.before_cut_stroke = pre_stroke_transits.count(c.stage) != 0U;
  return c;
}

/// Index of the last descent step: fractions 1 - kStep*step stay >= min_fraction (same tolerance as the loop).
inline int lastStep(double min_fraction)
{
  int step = 0;
  while (1.0 - kStep * (step + 1) >= min_fraction - 1e-9) ++step;
  return step;
}
inline double fractionOfStep(int step) { return 1.0 - kStep * step; }

enum class Action
{
  kStop,             // nothing left to try (the existing loop end)
  kTryNext,          // the next 5 % step
  kJumpToMinimum     // skip the steps in between, try the last step once
};
struct Decision
{
  Action action = Action::kStop;
  int next_step = 0;
};

/// What to do after the plan at `failed_step` failed. `allow_shortened`: top_cut_allow_shortened_cut.
/// `stop_on_independent`: the planning parameter; `before_cut_stroke`: the classification of this failure.
inline Decision decideAfterFailure(int failed_step, double min_fraction, bool allow_shortened, bool stop_on_independent, bool before_cut_stroke)
{
  Decision d;
  const int next = failed_step + 1;
  if (!allow_shortened || fractionOfStep(next) < min_fraction - 1e-9) return d;   // kStop
  const int last = lastStep(min_fraction);
  if (stop_on_independent && before_cut_stroke && next < last)
  {
    d.action = Action::kJumpToMinimum;
    d.next_step = last;
    return d;
  }
  d.action = Action::kTryNext;
  d.next_step = next;
  return d;
}

}  // namespace top_cut_descent
