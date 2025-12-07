#include "shared.hpp"
#include "sys/ioctl.h"
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

int get_terminal_width() {
  struct winsize w;
  if (ioctl(1, TIOCGWINSZ, &w) == -1 || w.ws_col == 0)
    return 80; // fallback
  return w.ws_col;
}

// Ex. prefix: "[CUDA] ". This will determine the space before each line.
void wrapped_print(const std::string& prefix, const std::string& text) {
  int width = get_terminal_width();
  int prefix_len = prefix.size();

  std::istringstream words(text);
  std::string word;

  int line_len = prefix_len;
  std::cout << prefix;

  while (words >> word) {
    if (line_len + 1 + (int)word.size() > width) {
      std::cout << "\n" << std::string(prefix_len, ' ');
      line_len = prefix_len;
    }

    if (line_len > prefix_len) {
      std::cout << " ";
      line_len++;
    }

    std::cout << word;
    line_len += word.size();
  }

  std::cout << "\n";
}
