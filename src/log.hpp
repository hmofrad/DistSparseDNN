/*
 * log.hpp: Logging tool
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef LOG_HPP
#define LOG_HPP

#include "env.hpp"

namespace Logging {
    bool enabled = false;
    const bool print_at_rank_zero = true;
    enum LOG_LEVEL {VOID, TRACE, DEBUG, INFO, WARN, ERROR, FATAL};
    const char* LOG_LEVELS[] = {"VOID", "TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    void print(int log_level, const char* format, ...);
}

void Logging::print(const int log_level, const char* format, ...) {
    if(enabled) {
        if((print_at_rank_zero and !Env::rank) or (not print_at_rank_zero)) {
            if(strncmp(Logging::LOG_LEVELS[log_level], LOG_LEVELS[VOID], 4)) {
                printf("%s[rank=%d] ", Logging::LOG_LEVELS[log_level], Env::rank);
            }
            va_list arglist;  
            va_start(arglist, format);
            vprintf(format, arglist);
            va_end(arglist);
        }
    }
}

#endif