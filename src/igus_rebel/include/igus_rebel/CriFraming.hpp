#ifndef IGUS_REBEL__CRI_FRAMING_HPP_
#define IGUS_REBEL__CRI_FRAMING_HPP_

#include <cstddef>
#include <string>
#include <vector>

namespace Igus
{
// Splits the CRI byte stream into message bodies.
//
// The robot sends "CRISTART <counter> <TYPE> ... CRIEND", with no guarantee that a TCP read
// ends on a message boundary. Feed() accepts arbitrary chunks and returns every complete
// body (the text between the markers, whitespace trimmed) in arrival order. Incomplete
// frames are kept (including their CRISTART marker) until the rest arrives. Bytes outside
// frames, empty bodies and frames truncated by a new CRISTART are discarded; the buffer is
// bounded by maxBufferedBytes. The class has no I/O and is not thread safe.
class CriFrameParser
{
public:
  static constexpr std::size_t kDefaultMaxBufferedBytes = 64 * 1024;

  explicit CriFrameParser(std::size_t maxBufferedBytes = kDefaultMaxBufferedBytes);

  std::vector<std::string> Feed(const char * data, std::size_t length);
  std::vector<std::string> Feed(const std::string & data);

  // Drops any partial frame (call on reconnect).
  void Reset();

  std::size_t BufferedBytes() const {return buffer_.size();}

  // Number of non-whitespace bytes discarded (garbage, truncated frames, overflow) since
  // the last call; resets the counter.
  std::size_t TakeDiscardedBytes();

private:
  void CountDiscarded(std::size_t begin, std::size_t end);

  std::string buffer_;
  std::size_t maxBufferedBytes_;
  std::size_t discardedBytes_ = 0;
};
}  // namespace Igus

#endif  // IGUS_REBEL__CRI_FRAMING_HPP_
