#ifndef IGUS_REBEL__FAULT_GROUP_HPP_
#define IGUS_REBEL__FAULT_GROUP_HPP_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>

// Process-wide "fault group" registry: several Igus::Rebel instances in one controller_manager
// that must stop together when one of them faults. No ROS dependency, no I/O.
//
// Lifetime: a FaultGroupMember is shared between its owner (the Rebel) and the registry, so the
// registry never holds a dangling pointer. The member only reaches its owner through the
// callback, and LeaveFaultGroup() blocks until a callback that is running has returned and then
// drops the callback, so an owner that leaves in its destructor (before its own members are
// destroyed) can never be called after destruction.
//
// Lock order (a total order, no cycles):
//   registry lock: held only while copying shared_ptrs / editing the map; it calls nothing and
//   is never held together with any other lock. Propagation copies the peer list under the
//   registry lock, releases it, and only then takes one peer's callbackLock_ at a time for the
//   duration of that peer's callback. The callback (Rebel::OnPeerFault) takes no mutex:
//   atomics and logging only. stateLock_ is a leaf (reason string), never held while calling
//   out. A Rebel never holds any of its own locks (snapshotLock, startupLock_, commandsLock,
//   cntLock) while calling into this registry, so the socket/message thread cannot be on the
//   wrong side of a cycle.
namespace Igus
{
class FaultGroupMember
{
public:
  // Called on the faulting thread for every other member of the group.
  using PeerFaultCallback =
    std::function<void (const std::string & originName, const std::string & reason)>;

  FaultGroupMember(std::string group, std::string name, PeerFaultCallback callback);

  const std::string & group() const {return group_;}
  const std::string & name() const {return name_;}

  // Root-cause bookkeeping. The first MarkRootFault() wins until ClearRootFault(); it takes
  // the next number of a process-wide sequence so that two members that both faulted on their
  // own are recovered in the order they faulted (no mutual blocking).
  void MarkRootFault(const std::string & reason);
  // Returns true if a root-cause fault was set.
  bool ClearRootFault();
  bool HasRootFault() const {return rootOrder_.load() != 0;}
  // 0 = no root-cause fault.
  std::uint64_t RootOrder() const {return rootOrder_.load();}
  std::string RootReason() const;
  bool Joined() const {return joined_.load();}

private:
  friend bool JoinFaultGroup(const std::shared_ptr<FaultGroupMember> &);
  friend void LeaveFaultGroup(const std::shared_ptr<FaultGroupMember> &);
  friend std::size_t ReportRootFault(
    const std::shared_ptr<FaultGroupMember> &, const std::string &);

  // Runs the callback (if still attached) while holding callbackLock_.
  bool Notify(const std::string & originName, const std::string & reason);
  void Detach();

  const std::string group_;
  const std::string name_;
  std::atomic<bool> joined_{false};
  std::atomic<std::uint64_t> rootOrder_{0};
  mutable std::mutex stateLock_;
  std::string rootReason_;
  std::mutex callbackLock_;
  PeerFaultCallback callback_;
};

using FaultGroupMemberPtr = std::shared_ptr<FaultGroupMember>;

// Registers the member in its group (no-op for an empty group name). If the member already
// holds a root-cause fault it is propagated to the existing members. Returns false if the
// member was not registered (no group, or already joined).
bool JoinFaultGroup(const FaultGroupMemberPtr & member);

// Removes the member from its group and detaches its callback (waits for a running callback).
// Idempotent. After it returns the member is never called back.
void LeaveFaultGroup(const FaultGroupMemberPtr & member);

// Marks `member` as root cause and calls the callback of every OTHER member of its group.
// Safe to call from any thread, repeatedly, and while peers are being destroyed. Returns the
// number of peers that were notified.
std::size_t ReportRootFault(const FaultGroupMemberPtr & member, const std::string & reason);

struct BlockingFault
{
  std::string member;
  std::string reason;
};

// Is another member of the group still holding an un-recovered root-cause fault that must be
// recovered before `member`? A non-root member is blocked by any other root-cause member; a
// root-cause member only by one that faulted before it.
bool FindBlockingRootFault(const FaultGroupMemberPtr & member, BlockingFault & blocking);

// Number of members currently registered in `group` (for tests).
std::size_t FaultGroupSize(const std::string & group);
}  // namespace Igus

#endif  // IGUS_REBEL__FAULT_GROUP_HPP_
