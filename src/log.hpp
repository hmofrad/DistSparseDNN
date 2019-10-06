/*
 * env.hpp: MPI runtime environment
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef LOGGING_HPP
#define LOGGING_HPP

namespace Logging {
    bool enabled = true;
    enum LOGLEVELS {TRACE, DEBUG, INFO, WARN, ERROR, FATAL};
    const char* LEVELS[] = {"TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"};
    void print(int this_rank, int log_level, const char* format, ...);
}

void Logging::print(int this_rank, int log_level, const char* format, ...) {
    if(enabled and (this_rank == 0)) {
        printf("%s: ", Logging::LEVELS[log_level]);
        va_list arglist;  
        va_start(arglist, format);
        vprintf(format, arglist);
        va_end(arglist);
    }
}

#endif