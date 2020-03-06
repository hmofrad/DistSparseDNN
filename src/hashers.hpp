/*
 * hashers.hpp: Hash functions implementation
 * (c) Mohammad Hasanzadeh Mofrad, 2019
 * (e) m.hasanzadeh.mofrad@gmail.com 
 */

/* Inspired from https://github.com/cmuq-ccl/LA3/blob/master/src/matrix/hashers.h */


#ifndef HASHERS_HPP
#define HASHERS_HPP

enum HASHING_TYPE {_EMPTY_, _BUCKET_};

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
        TwoDHasher(HASHING_TYPE hashing_type_, long nrows, long ncols) : hashing_type(hashing_type_) {
            if(hashing_type == HASHING_TYPE::_EMPTY_) {
                hasher_r = std::move(std::make_unique<NullHasher>());
                hasher_c = std::move(std::make_unique<NullHasher>());
                //hasher_x = new NullHasher();
                //hasher_y = new NullHasher();
            }
            else if(hashing_type == HASHING_TYPE::_BUCKET_) {
                hasher_r = std::move(std::make_unique<SimpleBucketHasher>(nrows, 1));
                hasher_c = std::move(std::make_unique<SimpleBucketHasher>(ncols, 1));
                //hasher_x = new SimpleBucketHasher(nrows, 1);
                //hasher_y = new SimpleBucketHasher(ncols, 1);
                
                //hasher_x = std::move(new SimpleBucketHasher(nrows, 1));
                //hasher_y = std::move(new SimpleBucketHasher(ncols, 1));
            }
        };
        
        ~TwoDHasher(){
            //delete(hasher_x);
            //delete(hasher_y);
        };
        HASHING_TYPE hashing_type;
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
