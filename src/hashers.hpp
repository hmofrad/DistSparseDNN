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
        
        //virtual std::pair<long, long> hash1(long x, long y);
        //virtual std::pair<long, long> unhash1(long x, long y);
};

class NullHasher : public ReversibleHasher {
    public:
        NullHasher() {}
        long   hash(long v) const {return v;}
        long unhash(long v) const {return v;}
};

class SimpleBucketHasher : public ReversibleHasher {
    private:
        const long multiplier = 1u; // For fine-granular load balance
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
        //TwoDHasher(HASHER_TYPE hasher_type_rows, HASHER_TYPE hasher_type_cols, long nrows, long ncols) {
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
            
            
            /*
            if(hashing_type_rows == HASHER_TYPE::_EMPTY_) {
                hasher_r = std::move(std::make_unique<NullHasher>());
            }
            else {
                hasher_r = std::move(std::make_unique<SimpleBucketHasher>(nrows, 1));
            }
            
            if(hashing_type_cols == HASHER_TYPE::_EMPTY_) {
                hasher_c = std::move(std::make_unique<NullHasher>());
            }
            else {
                hasher_c = std::move(std::make_unique<SimpleBucketHasher>(ncols, 1));
            }
            */
        };
        
        ~TwoDHasher(){
            //delete(hasher_x);
            //delete(hasher_y);
        };
        //HASHING_TYPE hashing_type;
        std::unique_ptr<ReversibleHasher> hasher_r = nullptr;
        std::unique_ptr<ReversibleHasher> hasher_c = nullptr;
        //ReversibleHasher* hasher_x = nullptr;
        //ReversibleHasher* hasher_y = nullptr;
        
};


/*
class TWODBucketHasher : public ReversibleHasher {
    private:
        const long nparts = 128u;
        long height = 0;
        long width = 0;
        long max_range_height = 0;
        long max_range_width = 0;

    public:
        TWODBucketHasher(long max_height, long max_width) {
            height = max_height / nparts;
            width = max_width / nparts;
            max_range_height = height * nparts;
            max_range_width = width * nparts;
        }

        std::pair<long, long> hash(long x, long y) {
            long r = 0;
            if(x >= max_range_height) r = x;
            long col = (uint32_t) x % nparts;
            long row = x / nparts;
            r = row + col * height;
            
            long c = 0;
            if(y >= max_range_width)  c = y;
            col = (uint32_t) y % nparts;
            row = y / nparts;
            c = row + col * height;
            
            return(std::make_pair(r, c));
        }

        std::pair<long, long> unhash(long x, long y) {
            long r = 0;
            if(x >= max_range_height) r = x;
            long col = x / height;
            long row = x % height;
            r = col + row * nparts;
            
            long c = 0;
            if(y >= max_range_height) c = y;
            col = y / height;
            row = y % height;
            c = col + row * nparts;
            
            return(std::make_pair(r, c));
        }
};
*/

#endif
