// -----------------------------------------------------------------------------
// tt.cpp
//
// Clustered transposition table. Each key maps (via multiply-high) to one 3-slot
// cluster; probes scan the cluster, stores update a matching slot in place or
// evict the least valuable one. "Least valuable" scores each slot by its search
// depth discounted for how many searches ago it was written, with a small hold
// for PV entries — so a deep result survives a shallow collision, but stale
// entries from previous searches are shed first.
// -----------------------------------------------------------------------------

#include "tt.hpp"

#include <algorithm>
#include <cstring>

namespace engine {

namespace {
// Replacement tuning. AGE_PENALTY is charged per generation of staleness, so
// within one `go` (all entries share the current age) replacement is purely
// depth-preferred, and across `go`s older entries lose ground. PV_HOLD keeps
// principal-variation entries around a touch longer. GEN_CYCLE matches the
// 5-bit generation field.
constexpr int AGE_PENALTY = 8;
constexpr int PV_HOLD     = 4;
constexpr int GEN_CYCLE   = 32;

// Mate scores are stored distance-to-mate from the *root*: normalise on write.
// (search.cpp holds the matching read-side conversion.)
Value value_to_tt(Value v, int ply) {
    if (v >= VALUE_MATE_IN_MAX_PLY) return v + ply;
    if (v <= VALUE_MATED_IN_MAX_PLY) return v - ply;
    return v;
}

[[nodiscard]] std::uint16_t tag_of(Key key) noexcept { return static_cast<std::uint16_t>(key); }
}  // namespace

void TranspositionTable::release() noexcept {
    if (clusters_) {
        ::operator delete(clusters_, std::align_val_t{64});
        clusters_ = nullptr;
    }
    cluster_count_ = 0;
}

void TranspositionTable::resize(std::size_t mb) {
    release();
    const std::size_t bytes = mb * 1024 * 1024;
    cluster_count_          = bytes / sizeof(Cluster);
    if (cluster_count_ < 1) cluster_count_ = 1;

    // Cache-line-aligned so no cluster straddles a line; raw allocation (entries
    // are trivial and made valid by the zero-fill in clear()).
    clusters_ = static_cast<Cluster*>(
        ::operator new(cluster_count_ * sizeof(Cluster), std::align_val_t{64}));
    clear();
}

void TranspositionTable::clear() {
    if (clusters_) std::memset(clusters_, 0, cluster_count_ * sizeof(Cluster));
    generation_ = 0;
}

int TranspositionTable::keep_priority(const TTEntry& e) const noexcept {
    if (e.bound() == Bound::NONE) return -(1 << 20);  // empty slot: evict first
    const int age = (generation_ - e.generation() + GEN_CYCLE) & (GEN_CYCLE - 1);
    return static_cast<int>(e.depth) - AGE_PENALTY * age + (e.is_pv() ? PV_HOLD : 0);
}

TTProbe TranspositionTable::probe(Key key) {
    if (cluster_count_ == 0) return {false, nullptr};

    Cluster&            c   = clusters_[cluster_index(key)];
    const std::uint16_t tag = tag_of(key);

    TTEntry* victim = &c.entry[0];
    int      worst  = keep_priority(c.entry[0]);
    for (int i = 0; i < 3; ++i) {
        TTEntry& e = c.entry[i];
        if (e.bound() != Bound::NONE && e.key16 == tag) return {true, &e};
        const int p = keep_priority(e);
        if (p < worst) { worst = p; victim = &e; }
    }
    return {false, victim};  // slot store() would overwrite on a miss
}

void TranspositionTable::store(Key key, Value value, Value eval, Bound bound, Depth depth, Move move,
                               int ply, bool pv) {
    if (cluster_count_ == 0) return;

    Cluster&            c   = clusters_[cluster_index(key)];
    const std::uint16_t tag = tag_of(key);

    // Target: a matching-key slot (update in place) or an empty one; otherwise the
    // lowest keep-priority slot in the cluster.
    TTEntry* tgt    = nullptr;
    TTEntry* victim = &c.entry[0];
    int      worst  = keep_priority(c.entry[0]);
    for (int i = 0; i < 3; ++i) {
        TTEntry& e = c.entry[i];
        if (e.bound() == Bound::NONE || e.key16 == tag) {
            tgt = &e;
            break;
        }
        const int p = keep_priority(e);
        if (p < worst) { worst = p; victim = &e; }
    }
    if (!tgt) tgt = victim;

    const bool same = tgt->bound() != Bound::NONE && tgt->key16 == tag;

    // Depth guard: don't replace a deeper same-position result with a shallower,
    // non-exact one — just refresh its age so it isn't shed as stale.
    if (same && bound != Bound::EXACT && depth + 3 < static_cast<Depth>(tgt->depth)) {
        tgt->gpb = static_cast<std::uint8_t>((generation_ << 3) | (tgt->is_pv() ? 0x4 : 0) |
                                             static_cast<std::uint8_t>(tgt->bound()));
        return;
    }

    // Preserve an existing best move if this store carries none for the position.
    const std::uint16_t mv = (move != Move(Move::NO_MOVE) || !same)
                                 ? static_cast<std::uint16_t>(move.move())
                                 : tgt->move16;
    const bool keep_pv = pv || (same && tgt->is_pv());  // PV-ness is sticky

    tgt->key16  = tag;
    tgt->move16 = mv;
    tgt->value  = static_cast<std::int16_t>(value_to_tt(value, ply));
    tgt->eval   = static_cast<std::int16_t>(eval);
    tgt->depth  = static_cast<std::uint8_t>(depth < 0 ? 0 : depth);
    tgt->gpb    = static_cast<std::uint8_t>((generation_ << 3) | (keep_pv ? 0x4 : 0) |
                                            static_cast<std::uint8_t>(bound));
}

int TranspositionTable::hashfull() const {
    if (cluster_count_ == 0) return 0;

    const std::size_t sample = std::min<std::size_t>(cluster_count_, 1000);
    int               used = 0, total = 0;
    for (std::size_t i = 0; i < sample; ++i)
        for (const TTEntry& e : clusters_[i].entry) {
            ++total;
            if (e.bound() != Bound::NONE && e.generation() == generation_) ++used;
        }
    return static_cast<int>(static_cast<std::size_t>(used) * 1000 / total);
}

}  // namespace engine
