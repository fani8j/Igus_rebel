#include "dual_cut_coordinator/plan_policy.hpp"

#include "dual_cut_coordinator/sequence_plan.hpp"   // shouldRetryPlan

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <utility>

#include <nlohmann/json.hpp>

namespace dual_cut
{
namespace
{
using json = nlohmann::ordered_json;

std::string esc(const std::string & in)
{
  std::string out;
  for (char c : in)
  {
    if (c == '"' || c == '\\') { out += '\\'; out += c; }
    else if (c == '\n') out += "\\n";
    else if (static_cast<unsigned char>(c) < 0x20) out += ' ';
    else out += c;
  }
  return out;
}

bool containsAny(const std::string & text, const std::vector<const char *> & needles)
{
  for (const char * n : needles)
    if (text.find(n) != std::string::npos) return true;
  return false;
}

std::string lower(std::string s)
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return s;
}
}  // namespace

// ---------------------------------------------------------------------------------------------
// chain
// ---------------------------------------------------------------------------------------------

const std::vector<std::string> & knownPlanVariantNames()
{
  static const std::vector<std::string> names = {"sequence", "sequence_left_top_first", "phases_top_first", "phases_side_first"};
  return names;
}

Status parsePlanVariant(const std::string & name, PlanVariant & out)
{
  out = PlanVariant{};
  out.name = name;
  if (name == "sequence") { out.sequence = true; out.left_top_first = false; return Status::Ok(); }
  if (name == "sequence_left_top_first") { out.sequence = true; out.left_top_first = true; return Status::Ok(); }
  if (name == "phases_top_first") { out.phase_order = "top_first"; return Status::Ok(); }
  if (name == "phases_side_first") { out.phase_order = "side_first"; return Status::Ok(); }
  return Status::Error(
    "plan_fallback_chain: unknown entry '" + name + "' (known: sequence, sequence_left_top_first, phases_top_first, phases_side_first)");
}

std::string configuredVariantName(const std::string & schedule_mode, const std::string & phase_order)
{
  return schedule_mode == "sequence" ? std::string("sequence") : "phases_" + phase_order;
}

Status buildPlanChain(const std::vector<std::string> & configured_chain, const std::string & configured_name, std::vector<PlanVariant> & out)
{
  out.clear();
  if (configured_chain.empty()) return Status::Error("plan_fallback_chain is empty");
  PlanVariant probe;
  for (size_t i = 0; i < configured_chain.size(); ++i)
  {
    Status s = parsePlanVariant(configured_chain[i], probe);
    if (!s) return s;
    for (size_t j = 0; j < i; ++j)
      if (configured_chain[j] == configured_chain[i]) return Status::Error("plan_fallback_chain: duplicate entry '" + configured_chain[i] + "'");
  }
  Status s = parsePlanVariant(configured_name, probe);
  if (!s) return Status::Error("configured mode: " + s.message);
  std::vector<std::string> names;
  const auto it = std::find(configured_chain.begin(), configured_chain.end(), configured_name);
  if (it != configured_chain.end())
  {
    names.assign(it, configured_chain.end());
  }
  else
  {
    names.push_back(configured_name);
    names.insert(names.end(), configured_chain.begin(), configured_chain.end());
  }
  for (const auto & n : names)
  {
    PlanVariant v;
    s = parsePlanVariant(n, v);
    if (!s) return s;
    out.push_back(v);
  }
  return Status::Ok();
}

const char * toString(RefusalClass c) { return c == RefusalClass::OrderFixable ? "order_fixable" : "not_order_fixable"; }

RefusalClass classifyRefusal(const std::string & m)
{
  // 1. never fixable by an order change (checked first)
  if (containsAny(
        m, {"precondition", "stop latched", "busy", "an execution is running", "execution is running", "holds the plan lock",
            "snapshot is stale", "stale-snapshot", "acquisition stamp", "snapshot rejected", "outside the detected carton top face",
            "top split source snapshot", "shorter than 50 mm", "source snapshot changed", "no external_snapshot_file",
            "frame-consistency", "planning_frame '", "tool_link '", "executor rejected parameter", "bad executor parameter",
            "must be 'phases' or 'sequence'", "must be 'top_first' or 'side_first'", "plan_fallback_chain", "planning scene",
            "carton '", "is not a single box", "expected the arm's base frame", "not available", "timeout calling",
            "timeout waiting", "timeout reading", "parameter service", "robot_description", "cannot parse",
            "varying orientation search failed closed", "executor plan mismatch", "retained executor plan"}))
    return RefusalClass::NotOrderFixable;
  // 2. executor plan-stage failure (the start state of the job / the phase structure changed with the order)
  if (m.find("plan_h_cut_from_state failed") != std::string::npos) return RefusalClass::OrderFixable;
  // 3. validator refusals of the planned trajectories and the sequence schedule search
  if (containsAny(
        m, {"world check", "tip-on-line", "cross-arm", "sequence schedule not possible", "top split coverage", "swept", "phase A refused", "phase B refused",
            "segment-window", "collision at sample"}))
    return RefusalClass::OrderFixable;
  return RefusalClass::NotOrderFixable;   // unknown text: do not guess
}

std::string planFallbackJson(
  const std::vector<PlanVariant> & chain, size_t current, const std::vector<PlanAttemptRecord> & previous, double budget_s,
  double elapsed_before_s, int pass)
{
  std::ostringstream o;
  o << std::setprecision(6) << "{\"chain\":[";
  for (size_t i = 0; i < chain.size(); ++i) o << (i ? "," : "") << "\"" << esc(chain[i].name) << "\"";
  o << "],\"attempt\":" << current + 1 << ",\"of\":" << chain.size() << ",\"variant\":\""
    << (current < chain.size() ? esc(chain[current].name) : std::string()) << "\",\"budget_s\":" << budget_s
    << ",\"elapsed_before_s\":" << elapsed_before_s << ",\"pass\":" << pass << ",\"previous\":[";
  for (size_t i = 0; i < previous.size(); ++i)
  {
    const auto & a = previous[i];
    o << (i ? "," : "") << "{\"variant\":\"" << esc(a.variant) << "\",\"reason\":\"" << esc(a.reason) << "\",\"class\":\"" << toString(a.cls)
      << "\",\"wall_s\":" << a.wall_s << ",\"retries\":" << a.retries << "}";
  }
  o << "]}";
  return o.str();
}

// ---------------------------------------------------------------------------------------------
// retry order
// ---------------------------------------------------------------------------------------------

bool retryInPlaceNow(const std::string & refusal, const PlanRetryPolicy & policy)
{
  if (!policy.breadth_first) return shouldRetryPlan(refusal, policy.signatures);   // former behaviour: every random-looking refusal
  if (!shouldRetryPlan(refusal, policy.signatures)) return false;
  for (const auto & label : policy.inplace_labels)
    if (!label.empty() && refusal.find(label) != std::string::npos) return true;
  return false;
}

PlanChainDriver::PlanChainDriver(size_t entries, PlanRetryPolicy policy)
: n_(entries), policy_(std::move(policy)), executions_(entries, 0), reasons_(entries), cls_(entries, RefusalClass::NotOrderFixable)
{
  if (n_ == 0) phase_ = Phase::Done;
  policy_.max_retries = std::max(0, policy_.max_retries);
}

bool PlanChainDriver::eligibleForSecondPass(size_t entry) const
{
  return executions_[entry] > 0 && !reasons_[entry].empty() && retriesUsed(entry) < policy_.max_retries &&
         shouldRetryPlan(reasons_[entry].back(), policy_.signatures);
}

PlanChainDriver::Next PlanChainDriver::next()
{
  Next out;
  if (phase_ == Phase::Done || awaiting_) return out;
  if (phase_ == Phase::First)
  {
    if (inplace_pending_)
    {
      inplace_pending_ = false;
      out = {false, cur_, true, false};
    }
    else if (first_next_ < n_)
    {
      cur_ = first_next_++;
      out = {false, cur_, false, false};
    }
    else if (policy_.breadth_first)
    {
      phase_ = Phase::Second;   // every entry was tried once and none passed: retry them in chain order
    }
    else
    {
      phase_ = Phase::Done;
      return out;
    }
    if (!out.done)
    {
      ++executions_[cur_];
      in_second_ = false;
      awaiting_ = true;
      return out;
    }
  }
  // second pass: round robin in chain order - every eligible entry gets its retry number r before any gets number r + 1
  bool found = false;
  size_t best = 0;
  for (size_t j = 0; j < n_; ++j)
  {
    if (!eligibleForSecondPass(j)) continue;
    if (!found || retriesUsed(j) < retriesUsed(best)) { best = j; found = true; }
  }
  if (found)
  {
    cur_ = best;
    ++executions_[best];
    in_second_ = true;
    awaiting_ = true;
    return Next{false, best, true, true};
  }
  phase_ = Phase::Done;
  return Next{};
}

void PlanChainDriver::succeeded()
{
  awaiting_ = false;
  ok_ = true;
  used_ = cur_;
  used_second_pass_ = in_second_;
  phase_ = Phase::Done;
}

void PlanChainDriver::failed(const std::string & reason, RefusalClass cls)
{
  awaiting_ = false;
  reasons_[cur_].push_back(reason);
  cls_[cur_] = cls;
  if (phase_ == Phase::First)
  {
    // a random-looking refusal on the in-place list is retried at once (the former behaviour retried all of them),
    // independent of its class, as before; otherwise a refusal no order change can fix ends the chain
    if (retriesUsed(cur_) < policy_.max_retries && retryInPlaceNow(reason, policy_))
    {
      inplace_pending_ = true;
      return;
    }
    if (cls == RefusalClass::NotOrderFixable) phase_ = Phase::Done;
    return;
  }
  // second pass: a refusal no order change can fix (stale snapshot, scene problem ...) makes every further retry pointless
  if (cls == RefusalClass::NotOrderFixable) phase_ = Phase::Done;
}

void PlanChainDriver::declineRetry()
{
  awaiting_ = false;
  --executions_[cur_];   // undo the execution counted by next()
  if (phase_ == Phase::First)
  {
    // the chain goes on (or ends for a refusal no order change can fix)
    inplace_pending_ = false;
    if (cls_[cur_] == RefusalClass::NotOrderFixable) phase_ = Phase::Done;
    return;
  }
  phase_ = Phase::Done;
}

void PlanChainDriver::stop()
{
  awaiting_ = false;
  phase_ = Phase::Done;
}

int PlanChainDriver::totalRetries() const
{
  int total = 0;
  for (size_t i = 0; i < n_; ++i) total += retriesUsed(i);
  return total;
}

std::string describeAttempts(const std::vector<PlanAttemptRecord> & attempts, size_t max_len)
{
  std::ostringstream o;
  for (size_t i = 0; i < attempts.size(); ++i)
  {
    std::string r = attempts[i].reason;
    if (r.size() > max_len) r = r.substr(0, max_len) + "...";
    o << (i ? "; " : "") << "[" << i + 1 << "] " << attempts[i].variant << ": " << r;
  }
  return o.str();
}

LeftJobOrder orderLeftJobs(
  bool top_first, const std::vector<std::string> & face, const std::vector<std::string> & top, const std::string & home_off,
  const std::string & home_on)
{
  LeftJobOrder out;
  out.top_first = top_first;
  const std::vector<std::string> & first = top_first ? top : face;
  const std::vector<std::string> & second = top_first ? face : top;
  out.job1 = first;
  out.job1.push_back(home_off);
  out.job2 = second;
  out.job2.push_back(home_on);
  out.job1_label = top_first ? "left top" : "left face";
  out.job2_label = top_first ? "left face" : "left top";
  return out;
}

// ---------------------------------------------------------------------------------------------
// detection permission
// ---------------------------------------------------------------------------------------------

const char * DetectionPermission::stateName() const
{
  switch (state)
  {
    case State::Allowed: return "allowed";
    case State::Blocked: return "blocked";
    default: return "unknown";
  }
}

bool overrideTruthy(const std::string & text)
{
  json v;
  try
  {
    v = json::parse(text);
  }
  catch (...)
  {
    return false;
  }
  if (v.is_null()) return false;
  if (v.is_boolean()) return v.get<bool>();
  if (v.is_number()) return v.get<double>() != 0.0;
  if (v.is_string())
  {
    const std::string s = lower(v.get<std::string>());
    return !(s.empty() || s == "false" || s == "none" || s == "no" || s == "off" || s == "0" || s == "null");
  }
  return !v.empty();   // array / object
}

DetectionPermission parseDetectionPermission(const std::string & snapshot_text)
{
  DetectionPermission p;
  json doc;
  try
  {
    doc = json::parse(snapshot_text);
  }
  catch (const std::exception & e)
  {
    p.state = DetectionPermission::State::Blocked;
    p.malformed = true;
    p.note = std::string("snapshot is not valid JSON: ") + e.what();
    return p;
  }
  if (doc.is_object())
  {
    if (doc.contains("snapshot_id") && doc["snapshot_id"].is_string()) p.snapshot_id = doc["snapshot_id"].get<std::string>();
    if (doc.contains("acquisition_stamp_ns") && doc["acquisition_stamp_ns"].is_number_unsigned())
      p.stamp_ns = doc["acquisition_stamp_ns"].get<uint64_t>();
  }
  if (!doc.is_object() || !doc.contains("cut_permission") || doc["cut_permission"].is_null())
  {
    p.state = DetectionPermission::State::Unknown;   // old snapshot without the field: allowed, said in the report
    return p;
  }
  const json & cp = doc["cut_permission"];
  if (!cp.is_object())
  {
    p.state = DetectionPermission::State::Blocked;
    p.malformed = true;
    p.note = "cut_permission is not an object";
    return p;
  }
  if (cp.contains("reasons") && cp["reasons"].is_array())
  {
    for (const auto & r : cp["reasons"])
    {
      if (r.is_object())
      {
        std::string code = r.contains("code") && r["code"].is_string() ? r["code"].get<std::string>() : r.dump();
        if (r.contains("detail") && r["detail"].is_string()) code += ": " + r["detail"].get<std::string>();
        p.reasons.push_back(code);
      }
      else if (r.is_string())
      {
        p.reasons.push_back(r.get<std::string>());
      }
      else
      {
        p.reasons.push_back(r.dump());
      }
    }
  }
  if (cp.contains("override") && !cp["override"].is_null())
  {
    p.override_text = cp["override"].dump();
    p.has_override = overrideTruthy(p.override_text);
  }
  const std::string status = cp.contains("status") && cp["status"].is_string() ? cp["status"].get<std::string>() : std::string();
  p.raw_status = status;
  if (status == "allowed")
  {
    p.state = DetectionPermission::State::Allowed;
  }
  else if (status == "blocked")
  {
    p.state = DetectionPermission::State::Blocked;
  }
  else
  {
    // present but unusable: fail closed
    p.state = DetectionPermission::State::Blocked;
    p.malformed = true;
    p.note = "cut_permission.status is '" + status + "' (expected allowed or blocked)";
  }
  return p;
}

Status checkPermissionMatchesPlan(const DetectionPermission & p, const std::string & plan_snapshot_id, uint64_t plan_stamp_ns)
{
  if (!p.snapshot_id.empty() && p.snapshot_id != plan_snapshot_id)
    return Status::Error(
      "source snapshot changed during planning: its detection permission was read from snapshot '" + p.snapshot_id + "' but the executors planned '" +
      plan_snapshot_id + "'");
  if (p.stamp_ns != 0 && p.stamp_ns != plan_stamp_ns)
    return Status::Error(
      "source snapshot changed during planning: acquisition_stamp_ns " + std::to_string(p.stamp_ns) + " (permission) != " + std::to_string(plan_stamp_ns) +
      " (plan)");
  return Status::Ok();
}

ExecuteGateResult detectionExecuteGate(const DetectionPermission & p, bool allow_blocked_detection, bool honor_override)
{
  ExecuteGateResult r;
  if (p.state == DetectionPermission::State::Allowed)
  {
    r.by = "allowed";
    return r;
  }
  if (p.state == DetectionPermission::State::Unknown)
  {
    r.by = "unknown_permission";
    return r;
  }
  std::string why = p.malformed ? ("cut_permission unusable (" + p.note + ")") : std::string("detection is blocked");
  for (const auto & s : p.reasons) why += "; " + s;
  if (allow_blocked_detection)
  {
    r.by = "allow_blocked_detection";
    r.message = "BLOCKED detection executed because allow_blocked_detection=true: " + why;
    return r;
  }
  if (honor_override && p.has_override && !p.malformed)
  {
    r.by = "detector_override";
    r.message = "BLOCKED detection executed on the detector's own override (" + p.override_text + "): " + why;
    return r;
  }
  r.allowed = false;
  r.by = "refused";
  r.message = why + " (set allow_blocked_detection:=true on /dual_cut_coordinator to execute anyway, or re-detect)";
  return r;
}

std::string detectionPermissionJson(const DetectionPermission & p, bool honor_override)
{
  std::ostringstream o;
  o << "{\"permission\":\"" << p.stateName() << "\",\"raw_status\":\"" << esc(p.raw_status) << "\",\"blocked\":"
    << (p.state == DetectionPermission::State::Blocked ? "true" : "false") << ",\"malformed\":" << (p.malformed ? "true" : "false")
    << ",\"note\":\"" << esc(p.note) << "\",\"reasons\":[";
  for (size_t i = 0; i < p.reasons.size(); ++i) o << (i ? "," : "") << "\"" << esc(p.reasons[i]) << "\"";
  o << "],\"override\":" << (p.override_text.empty() ? std::string("null") : p.override_text) << ",\"override_truthy\":"
    << (p.has_override ? "true" : "false") << ",\"override_honored\":" << (p.has_override && honor_override ? "true" : "false") << "}";
  return o.str();
}

}  // namespace dual_cut
