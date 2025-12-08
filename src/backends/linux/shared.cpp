#include "../../shared/shared.hpp"
#include <dlfcn.h>
#include <sys/ioctl.h>

void closeLibrary(void* handle) {
  if (handle) {
    dlclose(handle);
  }
}

int get_terminal_width() {
  struct winsize w;
  if (ioctl(1, TIOCGWINSZ, &w) == -1 || w.ws_col == 0)
    return 80; // fallback
  return w.ws_col;
}
