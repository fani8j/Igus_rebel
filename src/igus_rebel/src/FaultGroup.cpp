#include "igus_rebel/FaultGroup.hpp"

#include <algorithm>
#include <map>
#include <utility>
#include <vector>

namespace Igus
{
namespace
{
struct Registry
{
  std::mutex lock;
  std::map<std::string, std::vector<FaultGroupMemberPtr>> groups;
};

// Intentionally leaked: members may be left during static destruction of other objects.
Registry & GetRegistry()
{
  static Registry * registry = new Registry();
  return *registry;
}

std::atomic<std::uint64_t> g_nextRootOrder{1};

// Copy of the other members of the group, taken under the registry lock and used after it was
// released.
std::vector<FaultGroupMemberPtr> Peers(const FaultGroupMemberPtr & member)
{
  std::vector<FaultGroupMemberPtr> peers;
  Registry & registry = GetRegistry();
  std::lock_guard<std::mutex> guard(registry.lock);
  const auto it = registry.groups.find(member->group());
  if (it == registry.groups.end()) {
    return peers;
  }
  for (const auto & candidate : it->second) {
    if (candidate != member) {
      peers.push_back(candidate);
    }
  }
  return peers;
}
}  // namespace

FaultGroupMember::FaultGroupMember(
  std::string group, std::string name, PeerFaultCallback callback)
: group_(std::move(group)), name_(std::move(name)), callback_(std::move(callback))
{
}

void FaultGroupMember::MarkRootFault(const std::string & reason)
{
  if (rootOrder_.load() != 0) {
    return;
  }
  {
    std::lock_guard<std::mutex> guard(stateLock_);
    rootReason_ = reason;
  }
  std::uint64_t expected = 0;
  rootOrder_.compare_exchange_strong(expected, g_nextRootOrder.fetch_add(1));
}

bool FaultGroupMember::ClearRootFault()
{
  return rootOrder_.exchange(0) != 0;
}

std::string FaultGroupMember::RootReason() const
{
  std::lock_guard<std::mutex> guard(stateLock_);
  return rootReason_;
}

bool FaultGroupMember::Notify(const std::string & originName, const std::string & reason)
{
  std::lock_guard<std::mutex> guard(callbackLock_);
  if (!callback_) {
    return false;
  }
  callback_(originName, reason);
  return true;
}

void FaultGroupMember::Detach()
{
  std::lock_guard<std::mutex> guard(callbackLock_);
  callback_ = nullptr;
}

bool JoinFaultGroup(const FaultGroupMemberPtr & member)
{
  if (!member || member->group().empty()) {
    return false;
  }
  {
    Registry & registry = GetRegistry();
    std::lock_guard<std::mutex> guard(registry.lock);
    if (member->joined_.exchange(true)) {
      return false;
    }
    registry.groups[member->group()].push_back(member);
  }
  if (member->HasRootFault()) {
    ReportRootFault(member, member->RootReason());
  }
  return true;
}

void LeaveFaultGroup(const FaultGroupMemberPtr & member)
{
  if (!member) {
    return;
  }
  {
    Registry & registry = GetRegistry();
    std::lock_guard<std::mutex> guard(registry.lock);
    member->joined_ = false;
    const auto it = registry.groups.find(member->group());
    if (it != registry.groups.end()) {
      auto & members = it->second;
      members.erase(std::remove(members.begin(), members.end(), member), members.end());
      if (members.empty()) {
        registry.groups.erase(it);
      }
    }
  }
  // Registry lock released: wait for a callback that is running right now, then drop it.
  member->Detach();
}

std::size_t ReportRootFault(const FaultGroupMemberPtr & member, const std::string & reason)
{
  if (!member) {
    return 0;
  }
  member->MarkRootFault(reason);
  std::size_t notified = 0;
  for (const auto & peer : Peers(member)) {
    if (peer->Notify(member->name(), reason)) {
      ++notified;
    }
  }
  return notified;
}

bool FindBlockingRootFault(const FaultGroupMemberPtr & member, BlockingFault & blocking)
{
  if (!member) {
    return false;
  }
  const std::uint64_t own = member->RootOrder();
  for (const auto & peer : Peers(member)) {
    const std::uint64_t other = peer->RootOrder();
    if (other != 0 && (own == 0 || other < own)) {
      blocking.member = peer->name();
      blocking.reason = peer->RootReason();
      return true;
    }
  }
  return false;
}

std::size_t FaultGroupSize(const std::string & group)
{
  Registry & registry = GetRegistry();
  std::lock_guard<std::mutex> guard(registry.lock);
  const auto it = registry.groups.find(group);
  return it == registry.groups.end() ? 0 : it->second.size();
}
}  // namespace Igus
