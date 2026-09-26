#include "igus_rebel/CriFraming.hpp"

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

#include "igus_rebel/CriKeywords.hpp"

namespace Igus
{
namespace
{
bool IsSpace(char c)
{
  return std::isspace(static_cast<unsigned char>(c)) != 0;
}
}  // namespace

CriFrameParser::CriFrameParser(std::size_t maxBufferedBytes)
: maxBufferedBytes_(std::max<std::size_t>(maxBufferedBytes, CriKeywords::START.size()))
{
}

std::vector<std::string> CriFrameParser::Feed(const std::string & data)
{
  return Feed(data.data(), data.size());
}

std::vector<std::string> CriFrameParser::Feed(const char * data, std::size_t length)
{
  std::vector<std::string> bodies;
  if (data != nullptr && length > 0) {
    buffer_.append(data, length);
  }

  const std::string & startMarker = CriKeywords::START;
  const std::string & endMarker = CriKeywords::END;
  std::size_t pos = 0;

  while (true) {
    const std::size_t start = buffer_.find(startMarker, pos);
    if (start == std::string::npos) {
      // Keep the longest tail that is the beginning of a start marker split across reads.
      std::size_t keep = std::min(buffer_.size() - pos, startMarker.size() - 1);
      while (keep > 0 && buffer_.compare(buffer_.size() - keep, keep, startMarker, 0, keep) != 0) {
        --keep;
      }
      CountDiscarded(pos, buffer_.size() - keep);
      pos = buffer_.size() - keep;
      break;
    }

    CountDiscarded(pos, start);
    const std::size_t bodyStart = start + startMarker.size();
    const std::size_t end = buffer_.find(endMarker, bodyStart);
    const std::size_t nextStart = buffer_.find(startMarker, bodyStart);

    if (nextStart != std::string::npos && (end == std::string::npos || nextStart < end)) {
      // Truncated frame followed by a new frame: resynchronise on the new start marker.
      CountDiscarded(start, nextStart);
      pos = nextStart;
      continue;
    }

    if (end == std::string::npos) {
      // Incomplete frame: keep it, starting at its CRISTART marker, until more data arrives.
      pos = start;
      break;
    }

    std::size_t bodyBegin = bodyStart;
    std::size_t bodyEnd = end;
    while (bodyBegin < bodyEnd && IsSpace(buffer_[bodyBegin])) {
      ++bodyBegin;
    }
    while (bodyEnd > bodyBegin && IsSpace(buffer_[bodyEnd - 1])) {
      --bodyEnd;
    }
    if (bodyEnd > bodyBegin) {
      bodies.emplace_back(buffer_, bodyBegin, bodyEnd - bodyBegin);
    }
    pos = end + endMarker.size();
  }

  buffer_.erase(0, pos);

  if (buffer_.size() > maxBufferedBytes_) {
    CountDiscarded(0, buffer_.size());
    buffer_.clear();
  }

  return bodies;
}

void CriFrameParser::Reset()
{
  buffer_.clear();
}

std::size_t CriFrameParser::TakeDiscardedBytes()
{
  const std::size_t discarded = discardedBytes_;
  discardedBytes_ = 0;
  return discarded;
}

void CriFrameParser::CountDiscarded(std::size_t begin, std::size_t end)
{
  for (std::size_t i = begin; i < end && i < buffer_.size(); ++i) {
    if (!IsSpace(buffer_[i])) {
      ++discardedBytes_;
    }
  }
}
}  // namespace Igus
