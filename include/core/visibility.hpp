#pragma once

#if defined(_WIN32) || defined(_WIN64)
#    ifdef MAINERA_BUILD_DLL
#        define MAINERA_API __declspec(dllexport)
#    else
#        define MAINERA_API __declspec(dllimport)
#    endif
#else
#    define MAINERA_API __attribute__((visibility("default")))
#endif
