#pragma once

#include <string>
#include <string_view>

constexpr const static std::string_view RESET = "\033[0m";
constexpr const static std::string_view RED = "\033[31m";
constexpr const static std::string_view YELLOW = "\033[33m";
constexpr const static std::string_view GREEN = "\033[32m";
constexpr const static std::string_view BLUE = "\033[34m";
constexpr const static std::string_view CYAN = "\033[36m";
constexpr const static std::string_view MAGENTA = "\033[35m";
constexpr const static std::string_view CUDA = "\033[0m\033[32m[CUDA]\033[0m ";
constexpr const static std::string_view HIP = "\033[0m\033[34m[HIP]\033[0m ";
constexpr const static std::string_view ORCHESTRATOR = "\033[33m[GPUMark]\033[0m ";
constexpr const static std::string_view VULKAN = "\033[36m[Vulkan]\033[0m ";
constexpr const static std::string_view OPENCL = "\033[35m[OpenCL]\033[0m ";
constexpr const static std::string_view OPENGL = "\033[32m[OpenGL]\033[0m ";

std::string tolower(const std::string& str);
std::string toupper(const std::string& str);
std::string trim(const std::string& str);
std::string removeUnreadable(const std::string& str);
bool stringsRoughlyMatch(const std::string& a, const std::string& b);
int get_terminal_width();
void wrapped_print(const std::string& prefix, const std::string& text);
void closeLibrary(void* handle);