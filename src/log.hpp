/*
 * log.hpp: Logging tool
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef LOGGING_HPP
#define LOGGING_HPP

namespace Logging {
    bool enabled = false;
    bool print_at_rank_zero = true;
    enum LOGLEVELS {TRACE, DEBUG, INFO, WARN, ERROR, FATAL};
    const char* LEVELS[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    void print(int log_level, const char* format, ...);
}

void Logging::print(int log_level, const char* format, ...) {
    if(enabled) {
        if((print_at_rank_zero and !Env::rank) or (not print_at_rank_zero)) {
            printf("%s[rank=%d]: ", Logging::LEVELS[log_level], Env::rank);
            va_list arglist;  
            va_start(arglist, format);
            vprintf(format, arglist);
            va_end(arglist);
        }
    }
}

#endif