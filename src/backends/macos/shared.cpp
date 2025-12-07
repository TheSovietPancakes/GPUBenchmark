#include "../../shared/shared.hpp"
#include <dlfcn.h>

void closeLibrary(void* handle) {
  if (handle) {
    dlclose(handle);
  }
}
