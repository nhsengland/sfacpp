/*
    sfacpp - Estimation of TRE/GTRE SFA models
    Copyright (C) 2025 Edmund Haacke
    Copyright (C) 2025 NHS England

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include "utils/memoryusage.hpp"

#include <iostream>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__)
#include <mach/mach.h>
#elif defined(__linux__)
#include <unistd.h>
#include <sys/resource.h>
#include <fstream>
#endif

double memoryusage::get_memory_usage_mb() {
    size_t rss = 0;

#if defined(_WIN32)
    // Windows implementation
    PROCESS_MEMORY_COUNTERS info;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) {
        rss = info.WorkingSetSize;
    }

#elif defined(__APPLE__)
    // macOS implementation
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &infoCount) == KERN_SUCCESS) {
        rss = (size_t)info.resident_size;
    }

#elif defined(__linux__)
    // Linux implementation (reading from /proc/self/statm)
    // The first value is total size, second is resident set size (RSS)
    long pages = 0;
    std::ifstream ifs("/proc/self/statm");
    if (ifs >> pages >> pages) { // Read the second value
        rss = (size_t)pages * (size_t)sysconf(_SC_PAGESIZE);
    }
#endif

    return static_cast<double>(rss) / (1024.0 * 1024.0);
}