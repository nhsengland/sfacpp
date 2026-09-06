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

// #include "utils/ThreadContext.hpp"

#include "utils/ThreadContext.hpp"
#include <mutex>
#include <atomic>

// Global synchronization for key initialization
// static std::once_flag g_key_init_flag;
// static std::atomic<bool> g_key_valid{false};

// ==========================================
// Platform Specific Logic
// ==========================================

// #if defined(_WIN32)
//     #include <windows.h>
//     typedef DWORD ThreadKey;
//     static ThreadKey g_key = TLS_OUT_OF_INDEXES;

//     void platform_create_key_impl() {
//         g_key = TlsAlloc();
//         if (g_key != TLS_OUT_OF_INDEXES) {
//             g_key_valid = true;
//         }
//     }

//     void platform_delete_key() { 
//         if (g_key_valid) {
//             TlsFree(g_key);
//             g_key_valid = false;
//             g_key = TLS_OUT_OF_INDEXES;
//         }
//     }

//     void* platform_get() {
//         if (!g_key_valid) return nullptr;
//         return TlsGetValue(g_key);
//     }

//     void platform_set(void* p) {
//         if (g_key_valid) {
//             TlsSetValue(g_key, p);
//         }
//     }

// #else
//     #include <pthread.h>
//     typedef pthread_key_t ThreadKey;
//     static ThreadKey g_key;

//     // The Cleanup Callback (POSIX only)
//     // Runs automatically when a thread exits IF the value is not NULL
//     void cleanup_callback(void* ptr) {
//         if (ptr) {
//             delete static_cast<ThreadContext*>(ptr);
//         }
//     }

//     void platform_create_key_impl() {
//         // Create key with the cleanup callback for auto-destruction
//         if (pthread_key_create(&g_key, cleanup_callback) == 0) {
//             g_key_valid = true;
//         }
//     }

//     void platform_delete_key() { 
//         if (g_key_valid) {
//             pthread_key_delete(g_key);
//             g_key_valid = false;
//         }
//     }

//     void* platform_get() {
//         if (!g_key_valid) return nullptr;
//         return pthread_getspecific(g_key);
//     }

//     void platform_set(void* p) {
//         if (g_key_valid) {
//             pthread_setspecific(g_key, p);
//         }
//     }
// #endif

// ==========================================
// Public Interface
// ==========================================

void initializeThreadContext() {
    // std::call_once(g_key_init_flag, platform_create_key_impl);
}

// Ensure init happens if user forgot to call initializeThreadContext
void ensure_initialized() {
    // std::call_once(g_key_init_flag, platform_create_key_impl);
}

void freeCurrentThreadContext() {
    // // 1. Get the pointer
    // void* ptr = platform_get();
    
    // // 2. If it exists, delete it
    // if (ptr) {
    //     delete static_cast<ThreadContext*>(ptr);
    //     platform_set(nullptr); 
    // }
}

void teardownThreadContext() {
    // platform_delete_key();
    ThreadContext* ctx = getContext();
    if (!ctx) return;
    // reset all unique pointers
    ctx->wsGhkEstim.reset();
    ctx->treBaseJacHess.reset();
    ctx->treBaseWeightDens.reset();
    ctx->treDensHNorm.reset();
    ctx->treGradHessPanel.reset();
    ctx->treOpInner.reset();
    ctx->gtreEffPanel.reset();
    ctx->treEffPanel.reset();
    ctx->gtreDensPanel.reset();
    ctx->gtreGradHessPanel.reset();
}

ThreadContext* getContext()
{
    static thread_local ThreadContext ctx;
    return &ctx;
    // // Lazy init protection
    // ensure_initialized();

    // // If key creation failed (e.g. out of handles), return nullptr or handle error
    // if (!g_key_valid) return nullptr;

    // // Try to get existing struct
    // void* ptr = platform_get();

    // // If this thread has no context yet, allocate one
    // if (!ptr) {
    //     ThreadContext* newCtx = new ThreadContext();
    //     platform_set(newCtx);
    //     return newCtx;
    // }

    // return static_cast<ThreadContext*>(ptr);
}
