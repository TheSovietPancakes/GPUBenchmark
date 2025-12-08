#include "../../shared/shared.hpp"
#include <windows.h>

void closeLibrary(void* handle) {
  if (handle) {
    FreeLibrary(static_cast<HMODULE>(handle));
  }
}

int get_terminal_width() {
  CONSOLE_SCREEN_BUFFER_INFO csbi;
  int columns = 80; // Default width
  HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
  if (GetConsoleScreenBufferInfo(hStdout, &csbi)) {
    columns = csbi.srWindow.Right - csbi.srWindow.Left + 1;
  }
  return columns;
}
