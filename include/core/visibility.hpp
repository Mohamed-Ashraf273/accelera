#pragma once

#if defined(_WIN32) || defined(_WIN64)
#ifdef ACCELERA_BUILD_DLL
#define ACCELERA_API __declspec(dllexport)
#else
#define ACCELERA_API __declspec(dllimport)
#endif
#else
#define ACCELERA_API __attribute__((visibility("default")))
#endif
