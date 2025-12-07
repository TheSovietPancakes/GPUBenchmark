#include "../../shared/shared.hpp"
#include <windows.h>

void closeLibrary(void* handle) {
  if (handle) {
    FreeLibrary(static_cast<HMODULE>(handle));
  }
}
