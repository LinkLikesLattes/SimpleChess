#pragma once

// -----------------------------------------------------------------------------
// tt.hpp
//
// Transposition table. Entries are grouped into fixed 3-slot clusters sized to
// fit inside a single cache line, so one memory fetch delivers every candidate
// for a key and a collision can evict the least valuable slot instead of always
// clobbering a deep result. The table is indexed by a multiply-high of the key
// (so any byte size is usable, not just powers of two), the target cluster is
// prefetched a move ahead to hide the miss latency, and replacement scores each
// slot by depth discounted for staleness (with a small hold for PV entries).
// -----------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <new>

#include "types.hpp"

namespace engine {

// The kind of score stored in an entry, relative to the search window that
// produced it. EXACT is a PV score; LOWER/UPPER are fail-high/fail-low bounds.
enum class Bound : std::uint8_t {
    NONE  = 0,
    UPPER = 1,  // score is an upper bound (fail-low, alpha)
    LOWER = 2,  // score is a lower bound (fail-high, beta)
    EXACT = 3,  // score is exact (PV node)
};

// A single table slot (10 bytes). `key16` is the low 16 bits of the zobrist key
// as a collision tag — the index consumes the *high* bits via multiply-high, so
// the low bits are still discriminating here. `gpb` packs the search generation
// (5 bits), a PV-node flag (1 bit) and the Bound (2 bits) into one byte.
struct TTEntry {
    std::uint16_t key16  = 0;
    std::uint16_t move16 = 0;
    std::int16_t  value  = 0;  // score, root-relative on store
    std::int16_t  eval   = 0;  // static eval (for lazy re-use)
    std::uint8_t  depth  = 0;  // search depth this entry was made at
    std::uint8_t  gpb    = 0;  // generation<<3 | pv<<2 | bound

    [[nodiscard]] Bound bound() const noexcept { return static_cast<Bound>(gpb & 0x3); }
    [[nodiscard]] bool  is_pv() const noexcept { return (gpb & 0x4) != 0; }
    [[nodiscard]] std::uint8_t generation() const noexcept { return gpb >> 3; }
};

// Three entries per cluster, padded to exactly 32 bytes so a 64-byte-aligned
// table never splits a cluster across two cache lines.
struct Cluster {
    TTEntry     entry[3];
    std::uint8_t pad[2] = {0, 0};
};
static_assert(sizeof(TTEntry) == 10, "TTEntry must stay 10 bytes");
static_assert(sizeof(Cluster) == 32, "Cluster must fill half a cache line");

// Result of a probe: whether we hit, and a view of the slot we landed on (the
// matching entry on a hit, or the slot store() would overwrite on a miss).
struct TTProbe {
    bool     hit   = false;
    TTEntry* entry = nullptr;
};

class TranspositionTable {
   public:
    TranspositionTable() = default;
    ~TranspositionTable() { release(); }

    // (Re)allocate to `mb` megabytes. Uses the whole allocation (no power-of-two
    // rounding). Clears all entries. Called on `setoption Hash` and startup.
    void resize(std::size_t mb);

    // Zero every entry. Called on `ucinewgame`.
    void clear();

    // Advance the age counter so entries from earlier searches are discounted by
    // the replacement policy. Called once per `go`.
    void new_search() { generation_ = static_cast<std::uint8_t>((generation_ + 1) & 0x1F); }

    // Look up `key`. On a hit, `entry` points at the matching slot; on a miss it
    // points at the cluster slot store() would overwrite.
    [[nodiscard]] TTProbe probe(Key key);

    // Hint the memory system to pull the target cluster into cache. Issued a move
    // ahead of the probe so the miss overlaps with make-move / move generation.
    void prefetch(Key key) const noexcept {
        if (cluster_count_) __builtin_prefetch(&clusters_[cluster_index(key)]);
    }

    // Insert/replace an entry within its cluster.
    void store(Key key, Value value, Value eval, Bound bound, Depth depth, Move move, int ply,
               bool pv);

    // Rough fullness in permille (0..1000), for UCI `info hashfull`.
    [[nodiscard]] int hashfull() const;

   private:
    // Multiply-high: the top bits of key map uniformly onto [0, cluster_count_).
    [[nodiscard]] std::size_t cluster_index(Key key) const noexcept {
        return static_cast<std::size_t>(
            (static_cast<unsigned __int128>(key) * static_cast<unsigned __int128>(cluster_count_)) >>
            64);
    }

    // Keep-priority of a slot: higher survives, lower is the eviction victim.
    [[nodiscard]] int keep_priority(const TTEntry& e) const noexcept;

    void release() noexcept;

    Cluster*     clusters_      = nullptr;
    std::size_t  cluster_count_ = 0;
    std::uint8_t generation_    = 0;  // low 5 bits are the current search age
};

}  // namespace engine
