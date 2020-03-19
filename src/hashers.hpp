/*
 * hashers.hpp: Hash functions implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

/* Inspired from https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/hashers.h */


#ifndef HASHERS_HPP
#define HASHERS_HPP

enum HASHER_TYPE {_EMPTY_, _BUCKET_};
enum HASHING_TYPE {_NO_, _INPUT_, _LAYER_, _BOTH_};
const char* HASHING_TYPES[] = {"_NO_", "_INPUT_", "_LAYER_", "_BOTH_"};

class ReversibleHasher {
    public:
        virtual ~ReversibleHasher() {}
        virtual long hash(long v) const = 0;
        virtual long unhash(long v) const = 0;
};

class NullHasher : public ReversibleHasher {
    public:
        NullHasher() {}
        long   hash(long v) const {return v;}
        long unhash(long v) const {return v;}
};

class SimpleBucketHasher : public ReversibleHasher {
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


struct TwoDHasher {
    public:
        TwoDHasher(HASHING_TYPE hashing_type, bool is_input, long nrows, long ncols, long nbuckets_rows, long nbuckets_cols) {
            
            if(hashing_type == HASHING_TYPE::_NO_) {
                hasher_r = std::move(std::make_unique<NullHasher>());
                hasher_c = std::move(std::make_unique<NullHasher>());
            }
            else if(hashing_type == HASHING_TYPE::_INPUT_) {
                if(is_input) {
                    hasher_r = std::move(std::make_unique<SimpleBucketHasher>(nrows, nbuckets_rows));
                    hasher_c = std::move(std::make_unique<NullHasher>());
                }
                else {
                    hasher_r = std::move(std::make_unique<NullHasher>());
                    hasher_c = std::move(std::make_unique<NullHasher>());
                }
            }
            else if(hashing_type == HASHING_TYPE::_LAYER_) {
                if(is_input) {
                    hasher_r = std::move(std::make_unique<NullHasher>());
                    hasher_c = std::move(std::make_unique<SimpleBucketHasher>(ncols, nbuckets_cols));
                }
                else {
                    hasher_r = std::move(std::make_unique<SimpleBucketHasher>(nrows, nbuckets_rows));
                    hasher_c = std::move(std::make_unique<SimpleBucketHasher>(ncols, nbuckets_cols));
                }
            }
            else if(hashing_type == HASHING_TYPE::_BOTH_) {
                hasher_r = std::move(std::make_unique<SimpleBucketHasher>(nrows, nbuckets_rows));
                hasher_c = std::move(std::make_unique<SimpleBucketHasher>(ncols, nbuckets_cols));
            }
        };
        
        ~TwoDHasher(){}
        std::unique_ptr<ReversibleHasher> hasher_r = nullptr;
        std::unique_ptr<ReversibleHasher> hasher_c = nullptr;
};

#endif
