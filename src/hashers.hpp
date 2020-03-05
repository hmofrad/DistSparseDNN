/*
 * hashers.hpp: Hash functions implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

#ifndef HASHERS_HPP
#define HASHERS_HPP

enum Hashing_type
{
  _EMPTY_,
  _BUCKET_
};


/* Borrowed from LA3 https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/hashers.h */

class ReversibleHasher
{
    public:
        virtual ~ReversibleHasher() {}
        virtual long hash(long v) const = 0;
        virtual long unhash(long v) const = 0;
};

class NullHasher : public ReversibleHasher
{
    public:
        NullHasher() {}
        long hash(long v) const { return v; }
        long unhash(long v) const { return v; }
};

class SimpleBucketHasher : public ReversibleHasher
{
    private:
        const long multiplier = 128u; // For fine-granular load balance
        long nparts = 0;
        long height = 0;
        long max_range = 0;

    public:
        SimpleBucketHasher(long max_domain, long nbuckets)
        {
            nparts = nbuckets * multiplier;
            height = max_domain / nparts;
            max_range = height * nparts;
        }

        long hash(long v) const 
        {
            if(v >= max_range) return v;
            long col = (uint32_t) v % nparts;
            long row = v / nparts;
            return row + col * height;
        }

        long unhash(long v) const
        {
            if(v >= max_range) return v;
            long col = v / height;
            long row = v % height;
            return col + row * nparts;
        }
};


#endif
