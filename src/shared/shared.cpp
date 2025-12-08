#include "shared.hpp"
#include <cstdio>
#include <iostream>
#include <sstream>

std::string tolower(const std::string& str) {
  std::string lowerStr;
  lowerStr.reserve(str.size());
  for (unsigned char c : str) {
    lowerStr.push_back(std::tolower(c));
  }
  return lowerStr;
}

std::string toupper(const std::string& str) {
  std::string upperStr;
  upperStr.reserve(str.size());
  for (unsigned char c : str) {
    upperStr.push_back(std::toupper(c));
  }
  return upperStr;
}

std::string trim(const std::string& str) {
  const size_t first = str.find_first_not_of(" \t\n\r\f\v");
  if (first == std::string::npos) {
    return ""; // String is all whitespace
  }
  const size_t last = str.find_last_not_of(" \t\n\r\f\v");
  return str.substr(first, last - first + 1);
}

bool stringsRoughlyMatch(const std::string& a, const std::string& b) {
  std::string trimmedA = trim(a);
  std::string trimmedB = trim(b);
  return tolower(trimmedA) == tolower(trimmedB);
}

std::string removeUnreadable(const std::string& str) {
  std::string result;
  result.reserve(str.size());
  bool inEscape = false;
  for (size_t i = 0; i < str.size(); ++i) {
    if (!inEscape && str[i] == '\033' && i + 1 < str.size() && str[i + 1] == '[') {
      inEscape = true;
      ++i; // Skip the '['
      continue;
    }
    if (inEscape) {
      // Skip until we find a letter (ANSI sequence end)
      if ((str[i] >= 'A' && str[i] <= 'Z') || (str[i] >= 'a' && str[i] <= 'z')) {
        inEscape = false;
      }
      continue;
    }
    result.push_back(str[i]);
  }
  return result;
}

// Ex. prefix: "[CUDA] ". This will determine the space before each line.
// [CUDA] This is an
//        example of
//        wrapped text.
void wrapped_print(const std::string& prefix, const std::string& text) {
  int width = get_terminal_width();
  int prefix_len = removeUnreadable(prefix).size();

  std::istringstream lines(text);
  std::string line;
  bool first_line = true;
  while (std::getline(lines, line)) {
    std::istringstream words(line);
    std::string word;
    int current_len = prefix_len;
    if (first_line) {
      std::cout << prefix;
      first_line = false;
    } else {
      std::cout << "\n" << std::string(prefix_len, ' ');
      current_len = prefix_len;
    }
    bool first_word = true;
    while (words >> word) {
      int word_len = removeUnreadable(word).size();
      static const int space_len = 1;
      if (!first_word && current_len + word_len + space_len > width) {
        std::cout << "\n" << std::string(prefix_len, ' ');
        current_len = prefix_len;
        first_word = true;
      }
      if (!first_word) {
        std::cout << " ";
        current_len += space_len;
      }
      std::cout << word;
      current_len += word_len;
      first_word = false;
    }
  }
}
