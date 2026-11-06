// -----------------------------------------------------------------------------
// nnue.cpp — NNUE inference: threat-augmented HalfKA network, int8 SIMD play path.
//
// Architecture (matches train_bullet/src/train_threats.rs + threat_inputs.rs):
//   inputs = [ king-bucketed mirrored HalfKA-768 base (768*buckets)
//            | FullThreats threat features (59808)
//            | PP_3Wide pawn-pair features  (4560) ]
//   FT 512, crelu + pairwise-multiply; 8 material output buckets; body 512->L2->32->1.
//
// The threat / pawn-pair feature indexing here is byte-identical to the Rust
// trainer's threat_inputs.rs (validated by train_bullet's check_threats), so a
// net trained there evaluates identically here.
//
// Two paths: the float path (SCN4 nets, used for self-play labelling) and the
// quantised int8 SIMD path (SCN5 nets, the playing format), both driven by the
// incremental accumulator maintained around make/unmake.
//
// SCN4 float net (train_threats exporter):
//   "SCN4", u32{version=4, hl, input_buckets, l2, out_buckets},
//   then u32 len + f32[len] for each of l0w l0b l1w l1b l2w l2b l3w l3b.
//   total_inputs is derived as 768*input_buckets + 59808 + 4560.
//   (SCN4 float -> SCN5 int8 playing net via the net quantiser in the dev tooling.)
// -----------------------------------------------------------------------------

#include "nnue_internal.hpp"

namespace engine::nnue {

namespace {

// ---- incremental base accumulator (thread-local) ----------------------------
// Persistent per-perspective BASE accumulator (int16), maintained across
// make/unmake so the base sum isn't rebuilt every eval. Only the base block is
// incremental (Stage A); threats+pp are still recomputed in eval_quant.
constexpr int ACC_STACK = MAX_PLY + 8;
// The search never makes a move from a node at ply >= MAX_PLY (both negamax and qsearch return
// first), so the stack index peaks at MAX_PLY and the overflow branch in acc_make (which drops
// to full recompute for the rest of the iteration) is unreachable. Keep it that way.
static_assert(ACC_STACK > MAX_PLY + 1, "accumulator stack must cover every search ply");
struct Acc {
    alignas(64) std::int16_t v[2][MAX_HL];  // v[0]=WHITE perspective, v[1]=BLACK perspective
};

// Fixed-point (int32) accumulator stack for the float labeler path. Shares g_ply and
// the g_bb bitboard stack with the quant path; only one of g_stack / g_stack_fx is
// live per run (chosen by the loaded net format).
struct AccFx {
    alignas(64) std::int32_t v[2][MAX_HL];
};

// Absolute-perspective base feature index (persp 0=WHITE, 1=BLACK). color 0/1,
// t0 0..5 (pawn..king), king_abs = absolute king square of persp's side.
inline int base_feat(int persp, int color, int t0, int sq, int king_abs) {
    const int kp = (persp == 0) ? king_abs : (king_abs ^ 56);
    const int bucket = 768 * king_bucket(kp);
    const int flip = (file_of(kp) > 3) ? 7 : 0;
    const int rsq = (persp == 0) ? sq : (sq ^ 56);
    const int idx = ((color == persp) ? 0 : 384) + 64 * t0 + rsq;
    return bucket + (idx ^ flip);
}
// int16 += int8 with explicit NEON widening (the scalar/auto-vec form does not
// vectorize the mixed-width add). 16 int8/iter. hl must be a multiple of 16.
inline void acc_add(std::int16_t* a, const std::int8_t* w, int hl) {
#if defined(__ARM_NEON)
    for (int j = 0; j < hl; j += 16) {
        const int8x16_t wv = vld1q_s8(w + j);
        vst1q_s16(a + j, vaddq_s16(vld1q_s16(a + j), vmovl_s8(vget_low_s8(wv))));
        vst1q_s16(a + j + 8, vaddq_s16(vld1q_s16(a + j + 8), vmovl_s8(vget_high_s8(wv))));
    }
#else
    for (int j = 0; j < hl; ++j) a[j] += w[j];
#endif
}
[[maybe_unused]] inline void acc_sub(std::int16_t* a, const std::int8_t* w, int hl) {
#if defined(__ARM_NEON)
    for (int j = 0; j < hl; j += 16) {
        const int8x16_t wv = vld1q_s8(w + j);
        vst1q_s16(a + j, vsubq_s16(vld1q_s16(a + j), vmovl_s8(vget_low_s8(wv))));
        vst1q_s16(a + j + 8, vsubq_s16(vld1q_s16(a + j + 8), vmovl_s8(vget_high_s8(wv))));
    }
#else
    for (int j = 0; j < hl; ++j) a[j] -= w[j];
#endif
}

// Defined with the other fused kernels below; the refresh uses it for its one-pass form.
inline void fused_apply_src(std::int16_t* dst, const std::int16_t* src, const std::int8_t* const* add, int na,
                            const std::int8_t* const* sub, int ns, int hl);

// Full refresh (base + threats + pp) for a position, both perspectives. Uses
// gather routed to the absolute WHITE/BLACK-perspective accumulators.
void refresh_into(Acc& a, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl;
    const std::int8_t* l0 = n.l0w_i8.data();
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    // Collect every active feature column, then ONE src->dst fused pass per perspective
    // straight from the bias vector -- instead of a bias seed loop plus a 1 KB accumulator
    // read+write per feature. int16 sums are order-independent -> bit-identical.
    // (x86 campaign G1: +1.76%, all phases.)
    static thread_local const std::int8_t* cols[2][1024];
    int nc[2] = {0, 0};
    gather(
        board, n.base_dims,
        [&](std::size_t f) { cols[stmp][nc[stmp]++] = l0 + f * hl; },
        [&](std::size_t f) { cols[1 - stmp][nc[1 - stmp]++] = l0 + f * hl; });
    for (int p = 0; p < 2; ++p)
        fused_apply_src(a.v[p], n.l0b_i.data(), cols[p], nc[p], nullptr, 0, hl);
}

inline void fused_apply_src_fx(std::int32_t* dst, const std::int32_t* src, const std::int32_t* const* add, int na,
                               const std::int32_t* const* sub, int ns, int hl);

// Fixed-point (int32) full refresh for the float labeler path. Mirrors refresh_into:
// collect every active column, then ONE src->dst fused pass per perspective straight
// from the bias vector -- no bias seed loop, no 2 KB accumulator read+write per feature.
// int32 sums are order-independent -> bit-identical. (Self-play campaign FX1: +12.0%.)
void refresh_into_fx(AccFx& a, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl;
    const std::int32_t* l0 = n.l0w_fx_p;
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    static thread_local const std::int32_t* cols[2][1024];
    int nc[2] = {0, 0};
    gather(
        board, n.base_dims,
        [&](std::size_t f) { cols[stmp][nc[stmp]++] = l0 + f * hl; },
        [&](std::size_t f) { cols[1 - stmp][nc[1 - stmp]++] = l0 + f * hl; });
    for (int p = 0; p < 2; ++p)
        fused_apply_src_fx(a.v[p], n.l0b_fx.data(), cols[p], nc[p], nullptr, 0, hl);
}

// ---- incremental THREAT delta helpers ---------------------------------------
// Absolute board state in threat-index piece coding (sf = color<<3 | (type+1)); by[sf],
// occupancy, per-square piece, and king squares. Built from a chess::Board and
// mutated in place to form the after-move state.
struct BB {
    std::uint64_t occ = 0;
    std::uint64_t by[16] = {0};
    std::int8_t piece_on[64];
    int wk = 0, bk = 0;
    BB() = default;  // for the thread_local stack (slots are filled before use)
    explicit BB(const Board& b) {
        for (auto& p : piece_on) p = -1;
        std::uint64_t o = b.occ().getBits();
        while (o) {
            const int sq = __builtin_ctzll(o);
            o &= o - 1;
            const int pi = static_cast<int>(b.at(Square(sq)).internal());  // 0..11
            const int sf = ((pi / 6) << 3) | ((pi % 6) + 1);
            by[sf] |= 1ull << sq;
            piece_on[sq] = static_cast<std::int8_t>(sf);
        }
        occ = b.occ().getBits();
        wk = b.kingSq(Color::WHITE).index();
        bk = b.kingSq(Color::BLACK).index();
    }
};

// Per-ply incremental board bitboards, maintained in lockstep with the accumulator stack so a
// normal move reads its before-state from bb[ply] instead of rebuilding a BB from board.at()
// every make. Same ply / reset / make / unmake / null hooks.




}  // namespace

// Per-search-thread incremental state (accumulator stacks, bitboard stack, delta scratch). The
// search owns one per worker and passes it to every call, so the hot path makes no thread-local
// lookups (NPS 3.6 item C1: on macOS every thread_local access is a call through the TLV thunk).
struct Context {
    std::vector<Acc>   stack;     // int16 accumulators (quant net), one per ply
    std::vector<AccFx> stack_fx;  // int32 fixed-point accumulators (float labeler net)
    std::vector<BB>    bb;        // per-ply board bitboards, in lockstep with the stack
    int ply = -1;                 // -1 = uninitialised -> evaluate() recomputes from the board
    const void* addp[2][1024];    // delta column scratch (acc_make)
    const void* subp[2][1024];
    const void* ra[40];           // king-rebucket scratch
    const void* rs[40];
    // K1: per [perspective][mirror][bucket], the base-column sum of the pieces in `by` (as of the
    // last time this bucket was used).
    struct Finny {
        alignas(64) std::int16_t acc[MAX_HL];
        std::uint64_t by[16];
        bool valid = false;
    };
    Finny finny[2][2][kKingBuckets];
    unsigned finny_gen = 0;      // g_net_gen the caches were built for
    // evaluate()'s full-recompute fallback (no incremental accumulator yet). Kept here, not in
    // evaluate()'s frame: a frame over one page makes every call run the Windows stack probe.
    alignas(64) std::int16_t fb[2][MAX_HL];
    // FXZ: stack_fx[p] holds its values only when fx_ok[p]; otherwise fx_pend[p] has the column lists
    // acc_make recorded for it (adds in col[s][0, na), subs in col[s][na, na + ns)).
    // A rebucket (king crossed a bucket, same mirror) is left for that first read too. Only the low part
    // of each list is ever touched: no zeroing, no copies.
    struct FxPend {
        int kind;                       // 0: a normal move (the fields below), 1: a null move
        int from, to, mcolor, mt0, cpi;
        bool cap;
        int rb_color;                   // >= 0: a rebucket of that perspective is pending
        int na[2], ns[2];
#if !SC_FX_SCRATCH
        const std::int32_t* add[2][1024];
        const std::int32_t* sub[2][1024];
#endif
    };
    std::vector<std::uint8_t> fx_ok;
    std::unique_ptr<FxPend[]> fx_pend;
#if SC_FXP || SC_FXN
    AccFx fx_fb;   // FXP: the float eval's from-scratch fallback (kept out of its frame)
#endif
    // K1 for the float labeler: the int32 column sums, [perspective][mirror][bucket], built on first use.
    struct FinnyFx {
        alignas(64) std::int32_t acc[MAX_HL];
        std::uint64_t by[16];
        bool valid = false;
    };
    std::vector<FinnyFx> finny_fx;
    Key fx_root_key = 0;            // FXR: stack_fx[0] holds this position, for fx_root_gen's net
    unsigned fx_root_gen = 0;
    std::uint32_t fxd[2][2][1024];  // FXD: [before/after][perspective] feature lists of a special move
    std::vector<std::uint8_t> fxd_count;  // FXD: per feature, built on first use
#if SC_Q_LAZY
    // Lazy int8 updates: stack[p] holds its values only when q_ok[p]; otherwise q_pend[p] is the move to apply.
    struct QPend { std::int8_t kind, from, to, mcolor, mt0, cap, cpi, rb; };   // kind 1: a null move
    std::uint8_t q_ok[ACC_STACK];
    QPend q_pend[ACC_STACK];
#endif
};
// A pending ply's column lists: its own, or (SC_FX_SCRATCH) the context's scratch, since a ply's lists live
// only from its enumeration to the pass right after it.
using FxLists = const std::int32_t* [2][1024];
#if SC_FX_SCRATCH
inline FxLists& fx_adds(Context& cx, Context::FxPend&) { return *reinterpret_cast<FxLists*>(&cx.addp); }
inline FxLists& fx_subs(Context& cx, Context::FxPend&) { return *reinterpret_cast<FxLists*>(&cx.subp); }
#else
inline FxLists& fx_adds(Context&, Context::FxPend& e) { return e.add; }
inline FxLists& fx_subs(Context&, Context::FxPend& e) { return e.sub; }
#endif

// FXZ: bring stack_fx[p] up to date from the newest computed ancestor (fx_ok[0] is always set),
// one ply at a time, each the src->dst fused pass acc_make would have made.
void fx_enumerate(Context& cx, int k);
void fx_rebucket(Context& cx, Context::FxPend& e, const BB& bb1);

inline void fx_ensure(Context& cx, int p) {
    if (cx.fx_ok[p]) return;
    int q = p - 1;
    while (!cx.fx_ok[q]) --q;
    const int hl = g_net.hl;
    for (int k = q + 1; k <= p; ++k) {
        Context::FxPend& e = cx.fx_pend[k];
        if (e.kind == 0) fx_enumerate(cx, k);
        else cx.bb[k] = cx.bb[k - 1];   // a null move: same pieces, empty lists
        if (e.rb_color >= 0) fx_rebucket(cx, e, cx.bb[k]);
        for (int sd = 0; sd < 2; ++sd)
            fused_apply_src_fx(cx.stack_fx[k].v[sd], cx.stack_fx[k - 1].v[sd], fx_adds(cx, e)[sd], e.na[sd],
                               fx_subs(cx, e)[sd], e.ns[sd], hl);
        cx.fx_ok[k] = 1;
    }
}

#if SC_Q_LAZY
void q_apply(Context& cx, int k);
// Bring stack[p] up to date from the newest computed ancestor (q_ok[0] is always set), one ply at a time.
inline void q_ensure(Context& cx, int p) {
    if (cx.q_ok[p]) return;
    int q = p - 1;
    while (!cx.q_ok[q]) --q;
    for (int k = q + 1; k <= p; ++k) {
        q_apply(cx, k);
        cx.q_ok[k] = 1;
    }
}
#endif

// EVC: shared by the workers. A small table in static storage until eval_cache_resize, so there is always one.
std::uint64_t g_evc_boot[1 << 16];
std::uint64_t* g_evc = g_evc_boot;
std::size_t g_evc_mask = (1 << 16) - 1;

Context* context_new() { return new Context(); }
void context_free(Context* c) { delete c; }

namespace {
// The context of callers that do not own one (UCI debug commands): one per thread.
Context& tl_context() {
    thread_local Context c;
    return c;
}





// Apply many int8 weight columns to the int16 accumulator in ONE pass: read each
// 16-wide accumulator chunk once, fold in every add/sub column, write once. This
// amortizes the accumulator load/store traffic over all changed features instead
// of paying it per column (the per-column acc_add/acc_sub did read+write per
// feature). Correctness identical to sequential add/sub (order-independent sums).
inline void fused_apply(std::int16_t* acc, const std::int8_t* const* add, int na,
                        const std::int8_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    // Two independent accumulator lanes per chunk break the serial add-chain
    // dependency (better ILP across the M-series NEON units). int16 add is
    // associative mod 2^16, so any grouping is bit-identical.
    for (int c = 0; c < hl; c += 16) {
        int16x8_t lo = vld1q_s16(acc + c), hi = vld1q_s16(acc + c + 8);
        int16x8_t lo2 = vdupq_n_s16(0), hi2 = vdupq_n_s16(0);
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const int8x16_t w0 = vld1q_s8(add[k] + c), w1 = vld1q_s8(add[k + 1] + c);
            lo  = vaddq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vaddq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vaddq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vaddq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < na; ++k) {
            const int8x16_t w = vld1q_s8(add[k] + c);
            lo = vaddq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vaddq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const int8x16_t w0 = vld1q_s8(sub[k] + c), w1 = vld1q_s8(sub[k + 1] + c);
            lo  = vsubq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vsubq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vsubq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vsubq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < ns; ++k) {
            const int8x16_t w = vld1q_s8(sub[k] + c);
            lo = vsubq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vsubq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        vst1q_s16(acc + c,     vaddq_s16(lo, lo2));
        vst1q_s16(acc + c + 8, vaddq_s16(hi, hi2));
    }
#elif defined(__AVX2__)
    // AVX2 port of the NEON structure at 32 int16 per chunk (2 x 256-bit), two independent
    // accumulator lanes to break the add chain. int16 add is associative mod 2^16 -> any
    // grouping is bit-identical. hl must be a multiple of 32 (it is 512).
    // Few columns (quiet endgame makes): the fused chunk loop pays 16x the branchy tiny
    // k-loops (0-3 trips each, mispredicted); a column-outer pass with long predictable
    // 32-lane loops is cheaper. Same int16 sums -> bit-identical either way.
    // (x86 campaign A2c: +6.5% on <=6-piece endgames, -3.5% on a few middlegames, +0.44% mean.)
    // K=2: only 1-add/1-sub makes take the column-outer path (x86 campaign A2d; K=4 traded
    // -2.2% on every middlegame for +4.5% on endgames and failed the phase guard).
    constexpr int kSmallCols = 2;
    if (na + ns <= kSmallCols) {
        for (int k = 0; k < na; ++k) {
            const std::int8_t* w = add[k];
            for (int c = 0; c < hl; c += 32) {
                __m256i* p0 = reinterpret_cast<__m256i*>(acc + c);
                __m256i* p1 = reinterpret_cast<__m256i*>(acc + c + 16);
                _mm256_storeu_si256(p0, _mm256_add_epi16(_mm256_loadu_si256(p0),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c)))));
                _mm256_storeu_si256(p1, _mm256_add_epi16(_mm256_loadu_si256(p1),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 16)))));
            }
        }
        for (int k = 0; k < ns; ++k) {
            const std::int8_t* w = sub[k];
            for (int c = 0; c < hl; c += 32) {
                __m256i* p0 = reinterpret_cast<__m256i*>(acc + c);
                __m256i* p1 = reinterpret_cast<__m256i*>(acc + c + 16);
                _mm256_storeu_si256(p0, _mm256_sub_epi16(_mm256_loadu_si256(p0),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c)))));
                _mm256_storeu_si256(p1, _mm256_sub_epi16(_mm256_loadu_si256(p1),
                    _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + c + 16)))));
            }
        }
        return;
    }
    {
        auto w16 = [](const std::int8_t* p) {
            return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
        };
        int c = 0;
        for (; c + 64 <= hl; c += 64) {   // 64 int16 per chunk (4 vectors x 2 lanes): fewer iterations, more ILP (A9 +1.2%)
            __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 16));
            __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 32));
            __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 48));
            __m256i u0 = _mm256_setzero_si256(), u1 = u0, u2 = u0, u3 = u0;
            int k = 0;
            for (; k + 1 < na; k += 2) {
                const std::int8_t* p = add[k] + c; const std::int8_t* q = add[k + 1] + c;
                v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
                u0 = _mm256_add_epi16(u0, w16(q)); u1 = _mm256_add_epi16(u1, w16(q + 16)); u2 = _mm256_add_epi16(u2, w16(q + 32)); u3 = _mm256_add_epi16(u3, w16(q + 48));
            }
            for (; k < na; ++k) {
                const std::int8_t* p = add[k] + c;
                v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
            }
            for (k = 0; k + 1 < ns; k += 2) {
                const std::int8_t* p = sub[k] + c; const std::int8_t* q = sub[k + 1] + c;
                v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
                u0 = _mm256_sub_epi16(u0, w16(q)); u1 = _mm256_sub_epi16(u1, w16(q + 16)); u2 = _mm256_sub_epi16(u2, w16(q + 32)); u3 = _mm256_sub_epi16(u3, w16(q + 48));
            }
            for (; k < ns; ++k) {
                const std::int8_t* p = sub[k] + c;
                v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c),      _mm256_add_epi16(v0, u0));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 16), _mm256_add_epi16(v1, u1));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 32), _mm256_add_epi16(v2, u2));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 48), _mm256_add_epi16(v3, u3));
        }
        for (; c < hl; c += 32) {   // hl % 64 tail (empty for hl = 512)
            __m256i lo  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c));
            __m256i hi  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(acc + c + 16));
            __m256i lo2 = _mm256_setzero_si256(), hi2 = _mm256_setzero_si256();
            int k = 0;
            for (; k + 1 < na; k += 2) {
                lo  = _mm256_add_epi16(lo,  w16(add[k] + c));      hi  = _mm256_add_epi16(hi,  w16(add[k] + c + 16));
                lo2 = _mm256_add_epi16(lo2, w16(add[k + 1] + c));  hi2 = _mm256_add_epi16(hi2, w16(add[k + 1] + c + 16));
            }
            for (; k < na; ++k) {
                lo = _mm256_add_epi16(lo, w16(add[k] + c));  hi = _mm256_add_epi16(hi, w16(add[k] + c + 16));
            }
            for (k = 0; k + 1 < ns; k += 2) {
                lo  = _mm256_sub_epi16(lo,  w16(sub[k] + c));      hi  = _mm256_sub_epi16(hi,  w16(sub[k] + c + 16));
                lo2 = _mm256_sub_epi16(lo2, w16(sub[k + 1] + c));  hi2 = _mm256_sub_epi16(hi2, w16(sub[k + 1] + c + 16));
            }
            for (; k < ns; ++k) {
                lo = _mm256_sub_epi16(lo, w16(sub[k] + c));  hi = _mm256_sub_epi16(hi, w16(sub[k] + c + 16));
            }
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c),      _mm256_add_epi16(lo, lo2));
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(acc + c + 16), _mm256_add_epi16(hi, hi2));
        }
    }
#else
    for (int k = 0; k < na; ++k) acc_add(acc, add[k], hl);
    for (int k = 0; k < ns; ++k) acc_sub(acc, sub[k], hl);
#endif
}

// dst = src (+ add columns) (- sub columns) in ONE pass: reads src, writes dst. Replaces
// memcpy(dst, src) + fused_apply(dst, ...): the accumulator is read once and written once
// per chunk instead of copied, re-read and re-written. int16 sums are order-independent
// -> bit-identical (x86 campaign G4).
inline void fused_apply_src(std::int16_t* dst, const std::int16_t* src, const std::int8_t* const* add, int na,
                            const std::int8_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    // NEON src->dst form of fused_apply (NEON campaign N1): each 16-int16 chunk is read from
    // src once, every add/sub column is folded in over two independent lanes, and the chunk is
    // written to dst once -- no memcpy and no per-column accumulator pass (which is what the
    // #else fallback below costs on ARM). int16 add is associative mod 2^16 -> bit-identical.
    int c = 0;
    // N2b: 64 int16 per chunk -- 8 vectors x 2 lanes = 16 accumulators (a full FT row of a column
    // per two chunk iterations at hl = 512). Same wrapping int16 sums -> bit-identical.
#define SCN_COL64(V0, V1, V2, V3, V4, V5, V6, V7, P, OP)                                              \
    {                                                                                                  \
        const int8x16_t p0 = vld1q_s8((P) + c),      p1 = vld1q_s8((P) + c + 16),                      \
                        p2 = vld1q_s8((P) + c + 32), p3 = vld1q_s8((P) + c + 48);                      \
        V0 = OP(V0, vmovl_s8(vget_low_s8(p0)));  V1 = OP(V1, vmovl_s8(vget_high_s8(p0)));             \
        V2 = OP(V2, vmovl_s8(vget_low_s8(p1)));  V3 = OP(V3, vmovl_s8(vget_high_s8(p1)));             \
        V4 = OP(V4, vmovl_s8(vget_low_s8(p2)));  V5 = OP(V5, vmovl_s8(vget_high_s8(p2)));             \
        V6 = OP(V6, vmovl_s8(vget_low_s8(p3)));  V7 = OP(V7, vmovl_s8(vget_high_s8(p3)));             \
    }
    // A2 (NPS 3.6): 128 int16 per chunk, one set of 16 accumulators -- half the chunk iterations
    // and column-pointer reloads of the 64-lane form. Same wrapping int16 sums -> bit-identical.
    for (; c + 128 <= hl; c += 128) {
        int16x8_t v0 = vld1q_s16(src + c),       v1 = vld1q_s16(src + c + 8),
                  v2 = vld1q_s16(src + c + 16),  v3 = vld1q_s16(src + c + 24),
                  v4 = vld1q_s16(src + c + 32),  v5 = vld1q_s16(src + c + 40),
                  v6 = vld1q_s16(src + c + 48),  v7 = vld1q_s16(src + c + 56),
                  v8 = vld1q_s16(src + c + 64),  v9 = vld1q_s16(src + c + 72),
                  v10 = vld1q_s16(src + c + 80), v11 = vld1q_s16(src + c + 88),
                  v12 = vld1q_s16(src + c + 96), v13 = vld1q_s16(src + c + 104),
                  v14 = vld1q_s16(src + c + 112), v15 = vld1q_s16(src + c + 120);
        for (int k = 0; k < na; ++k) {
            const std::int8_t* P = add[k];   // SCN_COL64 adds the chunk offset c
            SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, P, vaddq_s16)
            SCN_COL64(v8, v9, v10, v11, v12, v13, v14, v15, P + 64, vaddq_s16)
        }
        for (int k = 0; k < ns; ++k) {
            const std::int8_t* P = sub[k];   // SCN_COL64 adds the chunk offset c
            SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, P, vsubq_s16)
            SCN_COL64(v8, v9, v10, v11, v12, v13, v14, v15, P + 64, vsubq_s16)
        }
        vst1q_s16(dst + c,       v0);  vst1q_s16(dst + c + 8,   v1);  vst1q_s16(dst + c + 16,  v2);
        vst1q_s16(dst + c + 24,  v3);  vst1q_s16(dst + c + 32,  v4);  vst1q_s16(dst + c + 40,  v5);
        vst1q_s16(dst + c + 48,  v6);  vst1q_s16(dst + c + 56,  v7);  vst1q_s16(dst + c + 64,  v8);
        vst1q_s16(dst + c + 72,  v9);  vst1q_s16(dst + c + 80,  v10); vst1q_s16(dst + c + 88,  v11);
        vst1q_s16(dst + c + 96,  v12); vst1q_s16(dst + c + 104, v13); vst1q_s16(dst + c + 112, v14);
        vst1q_s16(dst + c + 120, v15);
    }
    for (; c + 64 <= hl; c += 64) {
        int16x8_t v0 = vld1q_s16(src + c),      v1 = vld1q_s16(src + c + 8),
                  v2 = vld1q_s16(src + c + 16), v3 = vld1q_s16(src + c + 24),
                  v4 = vld1q_s16(src + c + 32), v5 = vld1q_s16(src + c + 40),
                  v6 = vld1q_s16(src + c + 48), v7 = vld1q_s16(src + c + 56);
        int16x8_t u0 = vdupq_n_s16(0), u1 = u0, u2 = u0, u3 = u0, u4 = u0, u5 = u0, u6 = u0, u7 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, add[k],     vaddq_s16)
            SCN_COL64(u0, u1, u2, u3, u4, u5, u6, u7, add[k + 1], vaddq_s16)
        }
        for (; k < na; ++k) SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, add[k], vaddq_s16)
        for (k = 0; k + 1 < ns; k += 2) {
            SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, sub[k],     vsubq_s16)
            SCN_COL64(u0, u1, u2, u3, u4, u5, u6, u7, sub[k + 1], vsubq_s16)
        }
        for (; k < ns; ++k) SCN_COL64(v0, v1, v2, v3, v4, v5, v6, v7, sub[k], vsubq_s16)
        vst1q_s16(dst + c,      vaddq_s16(v0, u0));  vst1q_s16(dst + c + 8,  vaddq_s16(v1, u1));
        vst1q_s16(dst + c + 16, vaddq_s16(v2, u2));  vst1q_s16(dst + c + 24, vaddq_s16(v3, u3));
        vst1q_s16(dst + c + 32, vaddq_s16(v4, u4));  vst1q_s16(dst + c + 40, vaddq_s16(v5, u5));
        vst1q_s16(dst + c + 48, vaddq_s16(v6, u6));  vst1q_s16(dst + c + 56, vaddq_s16(v7, u7));
    }
#undef SCN_COL64
    // N2 (A9-analog): 32 int16 per chunk -- 4 vectors x 2 lanes = 8 accumulators, half the
    // chunk iterations and twice the independent adds in flight. Same wrapping int16 sums.
    for (; c + 32 <= hl; c += 32) {
        int16x8_t v0 = vld1q_s16(src + c),      v1 = vld1q_s16(src + c + 8),
                  v2 = vld1q_s16(src + c + 16), v3 = vld1q_s16(src + c + 24);
        int16x8_t u0 = vdupq_n_s16(0), u1 = u0, u2 = u0, u3 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const int8x16_t p0 = vld1q_s8(add[k] + c),     p1 = vld1q_s8(add[k] + c + 16);
            const int8x16_t q0 = vld1q_s8(add[k + 1] + c), q1 = vld1q_s8(add[k + 1] + c + 16);
            v0 = vaddq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vaddq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vaddq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vaddq_s16(v3, vmovl_s8(vget_high_s8(p1)));
            u0 = vaddq_s16(u0, vmovl_s8(vget_low_s8(q0)));  u1 = vaddq_s16(u1, vmovl_s8(vget_high_s8(q0)));
            u2 = vaddq_s16(u2, vmovl_s8(vget_low_s8(q1)));  u3 = vaddq_s16(u3, vmovl_s8(vget_high_s8(q1)));
        }
        for (; k < na; ++k) {
            const int8x16_t p0 = vld1q_s8(add[k] + c), p1 = vld1q_s8(add[k] + c + 16);
            v0 = vaddq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vaddq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vaddq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vaddq_s16(v3, vmovl_s8(vget_high_s8(p1)));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const int8x16_t p0 = vld1q_s8(sub[k] + c),     p1 = vld1q_s8(sub[k] + c + 16);
            const int8x16_t q0 = vld1q_s8(sub[k + 1] + c), q1 = vld1q_s8(sub[k + 1] + c + 16);
            v0 = vsubq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vsubq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vsubq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vsubq_s16(v3, vmovl_s8(vget_high_s8(p1)));
            u0 = vsubq_s16(u0, vmovl_s8(vget_low_s8(q0)));  u1 = vsubq_s16(u1, vmovl_s8(vget_high_s8(q0)));
            u2 = vsubq_s16(u2, vmovl_s8(vget_low_s8(q1)));  u3 = vsubq_s16(u3, vmovl_s8(vget_high_s8(q1)));
        }
        for (; k < ns; ++k) {
            const int8x16_t p0 = vld1q_s8(sub[k] + c), p1 = vld1q_s8(sub[k] + c + 16);
            v0 = vsubq_s16(v0, vmovl_s8(vget_low_s8(p0)));  v1 = vsubq_s16(v1, vmovl_s8(vget_high_s8(p0)));
            v2 = vsubq_s16(v2, vmovl_s8(vget_low_s8(p1)));  v3 = vsubq_s16(v3, vmovl_s8(vget_high_s8(p1)));
        }
        vst1q_s16(dst + c,      vaddq_s16(v0, u0));  vst1q_s16(dst + c + 8,  vaddq_s16(v1, u1));
        vst1q_s16(dst + c + 16, vaddq_s16(v2, u2));  vst1q_s16(dst + c + 24, vaddq_s16(v3, u3));
    }
    for (; c < hl; c += 16) {   // 16-int16 chunks (the whole vector without N2; the hl % 32 tail with it)
        int16x8_t lo = vld1q_s16(src + c), hi = vld1q_s16(src + c + 8);
        int16x8_t lo2 = vdupq_n_s16(0), hi2 = vdupq_n_s16(0);
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const int8x16_t w0 = vld1q_s8(add[k] + c), w1 = vld1q_s8(add[k + 1] + c);
            lo  = vaddq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vaddq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vaddq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vaddq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < na; ++k) {
            const int8x16_t w = vld1q_s8(add[k] + c);
            lo = vaddq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vaddq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const int8x16_t w0 = vld1q_s8(sub[k] + c), w1 = vld1q_s8(sub[k + 1] + c);
            lo  = vsubq_s16(lo,  vmovl_s8(vget_low_s8(w0)));  hi  = vsubq_s16(hi,  vmovl_s8(vget_high_s8(w0)));
            lo2 = vsubq_s16(lo2, vmovl_s8(vget_low_s8(w1)));  hi2 = vsubq_s16(hi2, vmovl_s8(vget_high_s8(w1)));
        }
        for (; k < ns; ++k) {
            const int8x16_t w = vld1q_s8(sub[k] + c);
            lo = vsubq_s16(lo, vmovl_s8(vget_low_s8(w)));  hi = vsubq_s16(hi, vmovl_s8(vget_high_s8(w)));
        }
        vst1q_s16(dst + c,     vaddq_s16(lo, lo2));
        vst1q_s16(dst + c + 8, vaddq_s16(hi, hi2));
    }
#elif defined(__AVX2__)
    auto w16 = [](const std::int8_t* p) {
        return _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    };
    // A2d: makes with at most SC_NPS_SMALLCOLS columns go column-outer (long predictable loops).
    if (na + ns <= SC_NPS_SMALLCOLS) {
        const std::int16_t* s = src;
        for (int k = 0; k < na; ++k, s = dst) {
            const std::int8_t* w = add[k];
            for (int c = 0; c < hl; c += 32) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),
                    _mm256_add_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c)), w16(w + c)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16),
                    _mm256_add_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c + 16)), w16(w + c + 16)));
            }
        }
        for (int k = 0; k < ns; ++k, s = dst) {
            const std::int8_t* w = sub[k];
            for (int c = 0; c < hl; c += 32) {
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),
                    _mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c)), w16(w + c)));
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16),
                    _mm256_sub_epi16(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(s + c + 16)), w16(w + c + 16)));
            }
        }
        if (s == src) std::memcpy(dst, src, static_cast<std::size_t>(hl) * sizeof(std::int16_t));  // no columns at all
        return;
    }
    int c = 0;
    // A2 (NPS 3.6) on AVX2: 128 int16 per chunk, one set of 8 ymm accumulators -- half the chunk
    // iterations and column-pointer reloads of the 64-lane form. int16 adds wrap -> bit-identical.
    for (; c + 128 <= hl; c += 128) {
        auto ldv = [&](int o) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + o)); };
        __m256i v0 = ldv(0), v1 = ldv(16), v2 = ldv(32), v3 = ldv(48),
                v4 = ldv(64), v5 = ldv(80), v6 = ldv(96), v7 = ldv(112);
        for (int k = 0; k < na; ++k) {
            const std::int8_t* p = add[k] + c;
            v0 = _mm256_add_epi16(v0, w16(p));      v1 = _mm256_add_epi16(v1, w16(p + 16));
            v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
            v4 = _mm256_add_epi16(v4, w16(p + 64)); v5 = _mm256_add_epi16(v5, w16(p + 80));
            v6 = _mm256_add_epi16(v6, w16(p + 96)); v7 = _mm256_add_epi16(v7, w16(p + 112));
        }
        for (int k = 0; k < ns; ++k) {
            const std::int8_t* p = sub[k] + c;
            v0 = _mm256_sub_epi16(v0, w16(p));      v1 = _mm256_sub_epi16(v1, w16(p + 16));
            v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
            v4 = _mm256_sub_epi16(v4, w16(p + 64)); v5 = _mm256_sub_epi16(v5, w16(p + 80));
            v6 = _mm256_sub_epi16(v6, w16(p + 96)); v7 = _mm256_sub_epi16(v7, w16(p + 112));
        }
        auto stv = [&](int o, __m256i v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + o), v); };
        stv(0, v0); stv(16, v1); stv(32, v2); stv(48, v3); stv(64, v4); stv(80, v5); stv(96, v6); stv(112, v7);
    }
    for (; c + 64 <= hl; c += 64) {   // 64 int16 per chunk (4 vectors x 2 lanes), src -> dst (A9)
        __m256i v0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c));
        __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 16));
        __m256i v2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 32));
        __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 48));
        __m256i u0 = _mm256_setzero_si256(), u1 = u0, u2 = u0, u3 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            const std::int8_t* p = add[k] + c; const std::int8_t* q = add[k + 1] + c;
            v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
            u0 = _mm256_add_epi16(u0, w16(q)); u1 = _mm256_add_epi16(u1, w16(q + 16)); u2 = _mm256_add_epi16(u2, w16(q + 32)); u3 = _mm256_add_epi16(u3, w16(q + 48));
        }
        for (; k < na; ++k) {
            const std::int8_t* p = add[k] + c;
            v0 = _mm256_add_epi16(v0, w16(p)); v1 = _mm256_add_epi16(v1, w16(p + 16)); v2 = _mm256_add_epi16(v2, w16(p + 32)); v3 = _mm256_add_epi16(v3, w16(p + 48));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            const std::int8_t* p = sub[k] + c; const std::int8_t* q = sub[k + 1] + c;
            v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
            u0 = _mm256_sub_epi16(u0, w16(q)); u1 = _mm256_sub_epi16(u1, w16(q + 16)); u2 = _mm256_sub_epi16(u2, w16(q + 32)); u3 = _mm256_sub_epi16(u3, w16(q + 48));
        }
        for (; k < ns; ++k) {
            const std::int8_t* p = sub[k] + c;
            v0 = _mm256_sub_epi16(v0, w16(p)); v1 = _mm256_sub_epi16(v1, w16(p + 16)); v2 = _mm256_sub_epi16(v2, w16(p + 32)); v3 = _mm256_sub_epi16(v3, w16(p + 48));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),      _mm256_add_epi16(v0, u0));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16), _mm256_add_epi16(v1, u1));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 32), _mm256_add_epi16(v2, u2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 48), _mm256_add_epi16(v3, u3));
    }
    for (; c < hl; c += 32) {   // hl % 64 tail (empty for hl = 512)
        __m256i lo  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c));
        __m256i hi  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + c + 16));
        __m256i lo2 = _mm256_setzero_si256(), hi2 = _mm256_setzero_si256();
        int k = 0;
        for (; k + 1 < na; k += 2) {
            lo  = _mm256_add_epi16(lo,  w16(add[k] + c));      hi  = _mm256_add_epi16(hi,  w16(add[k] + c + 16));
            lo2 = _mm256_add_epi16(lo2, w16(add[k + 1] + c));  hi2 = _mm256_add_epi16(hi2, w16(add[k + 1] + c + 16));
        }
        for (; k < na; ++k) {
            lo = _mm256_add_epi16(lo, w16(add[k] + c));  hi = _mm256_add_epi16(hi, w16(add[k] + c + 16));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            lo  = _mm256_sub_epi16(lo,  w16(sub[k] + c));      hi  = _mm256_sub_epi16(hi,  w16(sub[k] + c + 16));
            lo2 = _mm256_sub_epi16(lo2, w16(sub[k + 1] + c));  hi2 = _mm256_sub_epi16(hi2, w16(sub[k + 1] + c + 16));
        }
        for (; k < ns; ++k) {
            lo = _mm256_sub_epi16(lo, w16(sub[k] + c));  hi = _mm256_sub_epi16(hi, w16(sub[k] + c + 16));
        }
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c),      _mm256_add_epi16(lo, lo2));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + c + 16), _mm256_add_epi16(hi, hi2));
    }
#else
    std::memcpy(dst, src, static_cast<std::size_t>(hl) * sizeof(std::int16_t));
    for (int k = 0; k < na; ++k) acc_add(dst, add[k], hl);
    for (int k = 0; k < ns; ++k) acc_sub(dst, sub[k], hl);
#endif
}

// Fixed-point in-place fused apply for the float labeler (int32 accumulator, int32 weight
// columns): the src == dst form of fused_apply_src_fx below (aliasing is safe: each chunk is
// fully loaded before it is stored). Integer add is associative -> bit-exact.
inline void fused_apply_fx(std::int32_t* acc, const std::int32_t* const* add, int na,
                           const std::int32_t* const* sub, int ns, int hl) {
    fused_apply_src_fx(acc, acc, add, na, sub, ns, hl);
}

// dst = src (+ add columns) (- sub columns) in ONE pass for the int32 fixed-point
// accumulator (float labeler): the FX analog of fused_apply_src (x86 G4 / NEON N1).
// Replaces memcpy(dst, src) + a separate column pass. int32 sums are order-independent
// -> bit-identical labels. (Self-play campaign FX2 single pass +6.5%, FX3 blocking +6.4%.)
inline void fused_apply_src_fx(std::int32_t* dst, const std::int32_t* src, const std::int32_t* const* add, int na,
                               const std::int32_t* const* sub, int ns, int hl) {
#if defined(__ARM_NEON)
    int c = 0;
    // N2b analog: 32 int32 per chunk -- 8 vectors x 2 lanes = 16 accumulators, a quarter of
    // the chunk iterations of the 8-wide form and 4x the adds in flight.
#define SCFX_COL32(V0, V1, V2, V3, V4, V5, V6, V7, P, OP)                                        \
    {                                                                                            \
        V0 = OP(V0, vld1q_s32((P) + c));      V1 = OP(V1, vld1q_s32((P) + c + 4));               \
        V2 = OP(V2, vld1q_s32((P) + c + 8));  V3 = OP(V3, vld1q_s32((P) + c + 12));              \
        V4 = OP(V4, vld1q_s32((P) + c + 16)); V5 = OP(V5, vld1q_s32((P) + c + 20));              \
        V6 = OP(V6, vld1q_s32((P) + c + 24)); V7 = OP(V7, vld1q_s32((P) + c + 28));              \
    }
    for (; c + 32 <= hl; c += 32) {
        int32x4_t v0 = vld1q_s32(src + c),      v1 = vld1q_s32(src + c + 4),
                  v2 = vld1q_s32(src + c + 8),  v3 = vld1q_s32(src + c + 12),
                  v4 = vld1q_s32(src + c + 16), v5 = vld1q_s32(src + c + 20),
                  v6 = vld1q_s32(src + c + 24), v7 = vld1q_s32(src + c + 28);
        int32x4_t u0 = vdupq_n_s32(0), u1 = u0, u2 = u0, u3 = u0, u4 = u0, u5 = u0, u6 = u0, u7 = u0;
        int k = 0;
        for (; k + 1 < na; k += 2) {
            SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, add[k],     vaddq_s32)
            SCFX_COL32(u0, u1, u2, u3, u4, u5, u6, u7, add[k + 1], vaddq_s32)
        }
        for (; k < na; ++k) SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, add[k], vaddq_s32)
        for (k = 0; k + 1 < ns; k += 2) {
            SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, sub[k],     vsubq_s32)
            SCFX_COL32(u0, u1, u2, u3, u4, u5, u6, u7, sub[k + 1], vsubq_s32)
        }
        for (; k < ns; ++k) SCFX_COL32(v0, v1, v2, v3, v4, v5, v6, v7, sub[k], vsubq_s32)
        vst1q_s32(dst + c,      vaddq_s32(v0, u0)); vst1q_s32(dst + c + 4,  vaddq_s32(v1, u1));
        vst1q_s32(dst + c + 8,  vaddq_s32(v2, u2)); vst1q_s32(dst + c + 12, vaddq_s32(v3, u3));
        vst1q_s32(dst + c + 16, vaddq_s32(v4, u4)); vst1q_s32(dst + c + 20, vaddq_s32(v5, u5));
        vst1q_s32(dst + c + 24, vaddq_s32(v6, u6)); vst1q_s32(dst + c + 28, vaddq_s32(v7, u7));
    }
#undef SCFX_COL32
    for (; c < hl; c += 8) {   // hl % 32 tail (empty for hl = 512)
        int32x4_t a0 = vld1q_s32(src + c),  a1 = vld1q_s32(src + c + 4);
        int32x4_t b0 = vdupq_n_s32(0),      b1 = vdupq_n_s32(0);
        int k = 0;
        for (; k + 1 < na; k += 2) {
            a0 = vaddq_s32(a0, vld1q_s32(add[k]     + c));  a1 = vaddq_s32(a1, vld1q_s32(add[k]     + c + 4));
            b0 = vaddq_s32(b0, vld1q_s32(add[k + 1] + c));  b1 = vaddq_s32(b1, vld1q_s32(add[k + 1] + c + 4));
        }
        for (; k < na; ++k) {
            a0 = vaddq_s32(a0, vld1q_s32(add[k] + c));      a1 = vaddq_s32(a1, vld1q_s32(add[k] + c + 4));
        }
        for (k = 0; k + 1 < ns; k += 2) {
            a0 = vsubq_s32(a0, vld1q_s32(sub[k]     + c));  a1 = vsubq_s32(a1, vld1q_s32(sub[k]     + c + 4));
            b0 = vsubq_s32(b0, vld1q_s32(sub[k + 1] + c));  b1 = vsubq_s32(b1, vld1q_s32(sub[k + 1] + c + 4));
        }
        for (; k < ns; ++k) {
            a0 = vsubq_s32(a0, vld1q_s32(sub[k] + c));      a1 = vsubq_s32(a1, vld1q_s32(sub[k] + c + 4));
        }
        vst1q_s32(dst + c,     vaddq_s32(a0, b0));
        vst1q_s32(dst + c + 4, vaddq_s32(a1, b1));
    }
#else
#if defined(__AVX2__)
    if constexpr (SC_FX_AVX2) {   // one src -> dst pass, 64 int32 (8 ymm) per chunk; int32 sums -> bit-exact
        auto ld = [](const std::int32_t* q) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q)); };
        auto st = [](std::int32_t* q, __m256i v) { _mm256_storeu_si256(reinterpret_cast<__m256i*>(q), v); };
        int c = 0;
        for (; c + 64 <= hl; c += 64) {
            __m256i v0 = ld(src + c),      v1 = ld(src + c + 8),  v2 = ld(src + c + 16), v3 = ld(src + c + 24),
                    v4 = ld(src + c + 32), v5 = ld(src + c + 40), v6 = ld(src + c + 48), v7 = ld(src + c + 56);
            for (int k = 0; k < na; ++k) {
                const std::int32_t* q = add[k] + c;
                v0 = _mm256_add_epi32(v0, ld(q));      v1 = _mm256_add_epi32(v1, ld(q + 8));
                v2 = _mm256_add_epi32(v2, ld(q + 16)); v3 = _mm256_add_epi32(v3, ld(q + 24));
                v4 = _mm256_add_epi32(v4, ld(q + 32)); v5 = _mm256_add_epi32(v5, ld(q + 40));
                v6 = _mm256_add_epi32(v6, ld(q + 48)); v7 = _mm256_add_epi32(v7, ld(q + 56));
            }
            for (int k = 0; k < ns; ++k) {
                const std::int32_t* q = sub[k] + c;
                v0 = _mm256_sub_epi32(v0, ld(q));      v1 = _mm256_sub_epi32(v1, ld(q + 8));
                v2 = _mm256_sub_epi32(v2, ld(q + 16)); v3 = _mm256_sub_epi32(v3, ld(q + 24));
                v4 = _mm256_sub_epi32(v4, ld(q + 32)); v5 = _mm256_sub_epi32(v5, ld(q + 40));
                v6 = _mm256_sub_epi32(v6, ld(q + 48)); v7 = _mm256_sub_epi32(v7, ld(q + 56));
            }
            st(dst + c, v0);      st(dst + c + 8, v1);  st(dst + c + 16, v2); st(dst + c + 24, v3);
            st(dst + c + 32, v4); st(dst + c + 40, v5); st(dst + c + 48, v6); st(dst + c + 56, v7);
        }
        for (; c < hl; c += 8) {   // hl % 64 tail (empty for hl = 512)
            __m256i v = ld(src + c);
            for (int k = 0; k < na; ++k) v = _mm256_add_epi32(v, ld(add[k] + c));
            for (int k = 0; k < ns; ++k) v = _mm256_sub_epi32(v, ld(sub[k] + c));
            st(dst + c, v);
        }
        return;
    }
#endif
    if (dst != src) std::memcpy(dst, src, static_cast<std::size_t>(hl) * sizeof(std::int32_t));
    for (int k = 0; k < na; ++k) { const std::int32_t* w = add[k]; for (int j = 0; j < hl; ++j) dst[j] += w[j]; }
    for (int k = 0; k < ns; ++k) { const std::int32_t* w = sub[k]; for (int j = 0; j < hl; ++j) dst[j] -= w[j]; }
#endif
}



// L2/L3 tail. Two numerically-close paths: a float matvec (bit-identical to the
// trainer/stock body) and an int8 sdot path (weights quantized at load; screlu
// activations quantized to int8 [0,127]). x1 is the float L1 output in [0,1].
Value finish_body(const Net& n, const float* x1, int b) {
    const int L2 = n.l2, OB = n.out_buckets;
    float y;
    std::array<float, 32> x2{};
    [[maybe_unused]] unsigned nz3 = 0xFFFFFFFFu;   // AVX2: the nonzero x2 (L3S)
    const float* l2w = n.l2w.data();
    const float* l2b = n.l2b.data() + b * 32;
    const std::size_t ostride = static_cast<std::size_t>(OB) * 32;
    (void)ostride;   // used by the scalar fallback only (NEON reads l2w_bm, AVX2 reads l2w_bm)
#if defined(__ARM_NEON)
    const float* w2base = l2w + static_cast<std::size_t>(b) * 32;  // row i at + i*ostride + o
    // N6 (G2 port): the bucket-major copy [(b*L2 + i)*32 + o] -- 4 KB contiguous per bucket, row i
    // at + i*32 -- instead of 32 rows at a 1 KB stride. Same values, same fma order -> bit-identical.
    const float* w2bm = n.l2w_bm.data() + static_cast<std::size_t>(b) * L2 * 32;
    (void)w2base;
    if (!g_l1dense && L2 == 32) {
        // §L3: NNZ sparse L2 (input-major over nonzero x1). screlu(x1) >= 0, ~half zero;
        // a skipped x1[i]==0 adds w*0.0f == 0 exactly -> per output the ascending-i fmla
        // chain is unchanged -> BYTE-IDENTICAL to the dense path. 8 f32x4 hold 32 outputs.
        int nz2[32]; int nnz2 = 0;
        for (int i = 0; i < 32; ++i) { nz2[nnz2] = i; nnz2 += (x1[i] != 0.0f); }
        float32x4_t c0=vld1q_f32(l2b),    c1=vld1q_f32(l2b+4),  c2=vld1q_f32(l2b+8),  c3=vld1q_f32(l2b+12),
                    c4=vld1q_f32(l2b+16), c5=vld1q_f32(l2b+20), c6=vld1q_f32(l2b+24), c7=vld1q_f32(l2b+28);
        for (int k = 0; k < nnz2; ++k) {
            const int i = nz2[k]; const float xi = x1[i];
            const float* w = w2bm + static_cast<std::size_t>(i) * 32;
            c0=vfmaq_n_f32(c0, vld1q_f32(w),    xi); c1=vfmaq_n_f32(c1, vld1q_f32(w+4),  xi);
            c2=vfmaq_n_f32(c2, vld1q_f32(w+8),  xi); c3=vfmaq_n_f32(c3, vld1q_f32(w+12), xi);
            c4=vfmaq_n_f32(c4, vld1q_f32(w+16), xi); c5=vfmaq_n_f32(c5, vld1q_f32(w+20), xi);
            c6=vfmaq_n_f32(c6, vld1q_f32(w+24), xi); c7=vfmaq_n_f32(c7, vld1q_f32(w+28), xi);
        }
        alignas(16) float tmp[32];
        vst1q_f32(tmp,c0);    vst1q_f32(tmp+4,c1);  vst1q_f32(tmp+8,c2);  vst1q_f32(tmp+12,c3);
        vst1q_f32(tmp+16,c4); vst1q_f32(tmp+20,c5); vst1q_f32(tmp+24,c6); vst1q_f32(tmp+28,c7);
        for (int o = 0; o < 32; ++o) x2[o] = screlu(tmp[o]);
    } else
    // 32x32 L2 matvec, 4 outputs at a time; per-output i-accumulation fmla chain,
    // matching the fp-contracted scalar -> bit-identical.
    for (int o = 0; o < 32; o += 4) {
        float32x4_t acc = vld1q_f32(l2b + o);
        for (int i = 0; i < L2; ++i)
            acc = vfmaq_n_f32(acc, vld1q_f32(w2bm + static_cast<std::size_t>(i) * 32 + o), x1[i]);
        float tmp[4];
        vst1q_f32(tmp, acc);
        for (int k = 0; k < 4; ++k) x2[o + k] = screlu(tmp[k]);
    }
#elif defined(__AVX2__)
    // AVX2: 4 x 8 outputs, i-outer. Per output the ascending-i FMA chain is exactly the
    // fp-contracted scalar chain (s = fma(x1[i], w, s)) -> bit-identical. Rows are 32
    // contiguous floats at l2w + i*ostride + b*32 (the scalar loop walks them with a
    // 1 KB stride per output instead).
    {
        const float* w2base = n.l2w_bm.data() + static_cast<std::size_t>(b) * L2 * 32;   // contiguous rows of 32 (G2)
        constexpr std::size_t wstride = 32;
        (void)l2w;
        __m256 c0 = _mm256_loadu_ps(l2b), c1 = _mm256_loadu_ps(l2b + 8),
               c2 = _mm256_loadu_ps(l2b + 16), c3 = _mm256_loadu_ps(l2b + 24);
        auto row = [&](int i) {
            const __m256 xi = _mm256_set1_ps(x1[i]);
            const float* w = w2base + static_cast<std::size_t>(i) * wstride;
            c0 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w),      c0);
            c1 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 8),  c1);
            c2 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 16), c2);
            c3 = _mm256_fmadd_ps(xi, _mm256_loadu_ps(w + 24), c3);
        };
        if (!g_l1dense && L2 == 32) {
            // NNZ-sparse L2 (the NEON §L3 form): only the nonzero x1 (screlu >= 0, about half are
            // zero), in ascending i, so every output's fma chain is the dense one minus terms that
            // add +0 -- the same value (a +-0 sum can only flip a zero's sign, which the screlu below
            // erases). Shorter chains: the dense loop is bound by fma latency, 32 deep.
            const __m256 z = _mm256_setzero_ps();
            unsigned m = 0;
            for (int q = 0; q < 4; ++q)
                m |= static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(_mm256_loadu_ps(x1 + 8 * q), z, _CMP_NEQ_OQ))) << (8 * q);
            while (m) { row(__builtin_ctz(m)); m &= m - 1; }
        } else {
            for (int i = 0; i < L2; ++i) row(i);
        }
        alignas(32) float tmp[32];
        _mm256_store_ps(tmp, c0); _mm256_store_ps(tmp + 8, c1);
        _mm256_store_ps(tmp + 16, c2); _mm256_store_ps(tmp + 24, c3);
        nz3 = 0;
        for (int o = 0; o < 32; o += 8) {   // 8-wide screlu, same ops per lane (clamp, square); A7
            __m256 v = _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(tmp + o), _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
            const __m256 sq = _mm256_mul_ps(v, v);
            _mm256_storeu_ps(x2.data() + o, sq);
            nz3 |= static_cast<unsigned>(_mm256_movemask_ps(_mm256_cmp_ps(sq, _mm256_setzero_ps(), _CMP_NEQ_OQ))) << o;
        }
    }
#else
    for (int o = 0; o < 32; ++o) {
        float s = l2b[o];
        for (int i = 0; i < L2; ++i)
            s += x1[i] * l2w[static_cast<std::size_t>(i) * ostride + b * 32 + o];
        x2[o] = screlu(s);
    }
#endif
    y = n.l3b[b];
    {   // contiguous per-bucket L3 row (A6); the sequential fma chain is unchanged -> bit-identical
        const float* w3 = n.l3w_t.data() + static_cast<std::size_t>(b) * 32;
#if defined(__AVX2__)
        // L3S: the same chain over the nonzero x2 only, ascending (dropped terms add +0: at most a
        // zero's sign changes, and lround maps +-0 alike). The dense chain is 32 dependent fma.
        if (!g_l1dense) {
            while (nz3) { const int i = __builtin_ctz(nz3); nz3 &= nz3 - 1; y += x2[i] * w3[i]; }
        } else
#endif
        for (int i = 0; i < 32; ++i) y += x2[i] * w3[i];
    }
    const float cp = std::clamp(400.0f * y, -15000.0f, 15000.0f);
    return static_cast<Value>(std::lround(cp));
}

// Float reference (matches the trainer's forward pass; parity-verified).
// §P1: build the ascending nonzero-index list of h[0..n) via NEON left-packing.
// Per 4 lanes, vcgtq_f32>0 -> 4-bit code -> a LUT of set-lane offsets (ascending),
// advance by popcount. Returns the SAME list as the branchless scalar build (so the
// sparse matvec is byte-identical); this just builds it faster.
#if defined(__ARM_NEON)
static const std::uint8_t kNnzLut[16][4] = {
    {0,0,0,0},{0,0,0,0},{1,0,0,0},{0,1,0,0},{2,0,0,0},{0,2,0,0},{1,2,0,0},{0,1,2,0},
    {3,0,0,0},{0,3,0,0},{1,3,0,0},{0,1,3,0},{2,3,0,0},{0,2,3,0},{1,2,3,0},{0,1,2,3}};
static const std::uint8_t kNnzPop[16] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
// S1: kNnzLut as uint16 offsets, for the sparse L1's chunk-index list.
alignas(8) static const std::uint16_t kNnzLut16[16][4] = {
    {0,0,0,0},{0,0,0,0},{1,0,0,0},{0,1,0,0},{2,0,0,0},{0,2,0,0},{1,2,0,0},{0,1,2,0},
    {3,0,0,0},{0,3,0,0},{1,3,0,0},{0,1,3,0},{2,3,0,0},{0,2,3,0},{1,2,3,0},{0,1,2,3}};
#endif
#if defined(__AVX2__)
// S1 on AVX2: for an 8-bit nonzero-chunk mask, the ascending set-bit offsets (one 128-bit store
// per 8 chunks, no data-dependent branch).
alignas(16) static constexpr auto kNnzLut8 = [] {
    std::array<std::array<std::uint16_t, 8>, 256> t{};
    for (int m = 0; m < 256; ++m) {
        int c = 0;
        for (int i = 0; i < 8; ++i)
            if ((m >> i) & 1) t[m][c++] = static_cast<std::uint16_t>(i);
    }
    return t;
}();
#endif

#if defined(__ARM_NEON)
inline int build_nnz(const float* h, int n, int* nz) {
    int nnz = 0;
    const float32x4_t z = vdupq_n_f32(0.0f);
    const uint32x4_t  bit = {1u, 2u, 4u, 8u};
    for (int i = 0; i < n; i += 4) {
        const uint32x4_t m = vcgtq_f32(vld1q_f32(h + i), z);
        const unsigned code = vaddvq_u32(vandq_u32(m, bit));  // 0..15
        const std::uint8_t* off = kNnzLut[code];
        nz[nnz + 0] = i + off[0];  nz[nnz + 1] = i + off[1];
        nz[nnz + 2] = i + off[2];  nz[nnz + 3] = i + off[3];
        nnz += kNnzPop[code];
    }
    return nnz;
}
#endif

// SWAP=false: the position as it is. SWAP=true: the same accumulator read with the two
// perspectives exchanged (the position after a pass), negated so the result is from the
// ORIGINAL side to move's point of view. See evaluate_pass() in nnue.hpp.
#if SC_FXP || SC_FXN
template <bool SWAP>
[[gnu::noinline]] Value eval_float_t0(Context& cx, const Board& board);

[[gnu::noinline, gnu::cold]] void fx_verify_now(const AccFx& a, int stmp, const Board& board, int ply) {
    AccFx chk;
    refresh_into_fx(chk, board);
    for (int j = 0; j < g_net.hl; ++j)
        if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
            std::fprintf(stderr, "[nnue] FX ACC MISMATCH ply=%d j=%d inc=%d,%d chk=%d,%d\n",
                         ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
            break;
        }
}
#endif

#if SC_FXP
template <bool SWAP>
[[gnu::noinline]] Value eval_float_t(Context& cx, const Board& board) {   // its buffers stay out of evaluate()'s frame
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    if (g_l1dense || L2 != 32 || (hl & 63) != 0) return eval_float_t0<SWAP>(cx, board);
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    const AccFx* ap;
    if (cx.ply >= 0) {  // search has the incremental accumulator in sync
        if constexpr (SC_FX_LAZY) fx_ensure(cx, cx.ply);
        ap = &cx.stack_fx[cx.ply];
        if (g_verify) fx_verify_now(*ap, stmp, board, cx.ply);   // incremental must equal a fresh refresh
    } else {  // outside search (bare `eval`): from-scratch fixed-point
        refresh_into_fx(cx.fx_fb, board);
        ap = &cx.fx_fb;
    }
    // h = crelu(a) * crelu(b) of (float)v * 2^-22 (the FXA fill and pairwise ops, lane for lane), and
    // per 64 inputs the mask of nonzero h for the L1. SWAP exchanges the two perspectives' halves.
    const int off_stm = SWAP ? half : 0;
    const int off_ntm = SWAP ? 0 : half;
    alignas(64) float h[MAX_HL];
    std::uint64_t hm[MAX_HL / 64] = {};
    {
        const __m256 inv = _mm256_set1_ps(float(1.0 / FX_S)), vz = _mm256_setzero_ps(), vo = _mm256_set1_ps(1.0f);
        auto cl = [&](const std::int32_t* q) {
            return _mm256_min_ps(_mm256_max_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(
                       _mm256_loadu_si256(reinterpret_cast<const __m256i*>(q))), inv), vz), vo);
        };
        const std::int32_t* vs = ap->v[stmp];
        const std::int32_t* vn = ap->v[1 - stmp];
        for (int i = 0; i < half; i += 8) {
            const __m256 ps = _mm256_mul_ps(cl(vs + i), cl(vs + i + half));
            const __m256 pn = _mm256_mul_ps(cl(vn + i), cl(vn + i + half));
            _mm256_store_ps(h + off_stm + i, ps);
            _mm256_store_ps(h + off_ntm + i, pn);
            hm[(off_stm + i) >> 6] |= static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_cmp_ps(ps, vz, _CMP_NEQ_OQ)))) << ((off_stm + i) & 63);
            hm[(off_ntm + i) >> 6] |= static_cast<std::uint64_t>(static_cast<unsigned>(
                _mm256_movemask_ps(_mm256_cmp_ps(pn, vz, _CMP_NEQ_OQ)))) << ((off_ntm + i) & 63);
        }
    }
    const int b = n.obucket[board.occ().count()];
    float x1[MAX_L2];
    {   // the FXA/FXB L1: input-major over the nonzero h[i] ascending, all 32 outputs in 4 ymm
        const bool bm = !n.l1w_bm.empty();
        const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
        const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                                : &n.l1w[static_cast<std::size_t>(b) * L2];
        const float* l1bb  = &n.l1b[b * L2];
        __m256 a0 = _mm256_loadu_ps(l1bb), a1 = _mm256_loadu_ps(l1bb + 8),
               a2 = _mm256_loadu_ps(l1bb + 16), a3 = _mm256_loadu_ps(l1bb + 24);
        for (int g = 0; g < hl; g += 64) {
            std::uint64_t m = hm[g >> 6];
            while (m) {
                const int i = g + __builtin_ctzll(m);
                m &= m - 1;
                const __m256 hv = _mm256_set1_ps(h[i]);
                const float* w = wbase + static_cast<std::size_t>(i) * istride;
                a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w),      a0);
                a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 8),  a1);
                a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 16), a2);
                a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 24), a3);
            }
        }
        const __m256 vz = _mm256_setzero_ps(), vo = _mm256_set1_ps(1.0f);
        const __m256 acc[4] = {a0, a1, a2, a3};
        for (int q = 0; q < 4; ++q) {   // 8-wide screlu, same ops per lane (clamp, square)
            const __m256 v = _mm256_min_ps(_mm256_max_ps(acc[q], vz), vo);
            _mm256_storeu_ps(x1 + 8 * q, _mm256_mul_ps(v, v));
        }
    }
    const Value v = finish_body(n, x1, b);
    return SWAP ? -v : v;
}
#endif

#if SC_FXN
// vqtbl1q_u8 masks: the 4-byte lanes set in a 4-bit code, moved to the front in ascending order.
constexpr auto kFxnPack = [] {
    std::array<std::array<std::uint8_t, 16>, 16> t{};
    for (int code = 0; code < 16; ++code) {
        int out = 0;
        for (int l = 0; l < 4; ++l)
            if ((code >> l) & 1) {
                for (int b = 0; b < 4; ++b) t[code][out * 4 + b] = static_cast<std::uint8_t>(l * 4 + b);
                ++out;
            }
        for (int b = out * 4; b < 16; ++b) t[code][b] = 0xFF;
    }
    return t;
}();

template <bool SWAP>
[[gnu::noinline]] Value eval_float_t(Context& cx, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    if (g_l1dense || L2 != 32 || (hl & 63) != 0) return eval_float_t0<SWAP>(cx, board);
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    const AccFx* ap;
    if (cx.ply >= 0) {  // search has the incremental accumulator in sync
        if constexpr (SC_FX_LAZY) fx_ensure(cx, cx.ply);
        ap = &cx.stack_fx[cx.ply];
        if (g_verify) fx_verify_now(*ap, stmp, board, cx.ply);   // incremental must equal a fresh refresh
    } else {  // outside search (bare `eval`): from-scratch fixed-point
        refresh_into_fx(cx.fx_fb, board);
        ap = &cx.fx_fb;
    }
    const int b = n.obucket[board.occ().count()];
    const bool bm = !n.l1w_bm.empty();
    const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
    const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                            : &n.l1w[static_cast<std::size_t>(b) * L2];
    const float* l1bb  = &n.l1b[b * L2];
    float32x4_t a0=vld1q_f32(l1bb),    a1=vld1q_f32(l1bb+4),  a2=vld1q_f32(l1bb+8),
                a3=vld1q_f32(l1bb+12), a4=vld1q_f32(l1bb+16), a5=vld1q_f32(l1bb+20),
                a6=vld1q_f32(l1bb+24), a7=vld1q_f32(l1bb+28);
    const float32x4_t inv = vdupq_n_f32(float(1.0 / FX_S)), vz = vdupq_n_f32(0.0f), vo = vdupq_n_f32(1.0f);
    const int32x4_t   iz  = vdupq_n_s32(0);
    // h[0, half) comes from the first perspective, h[half, hl) from the other; SWAP exchanges them.
    // Pack the inputs of the nonzero h[i] (ascending i, branchless), then the float ops on those only.
    const std::int32_t* const blocks[2] = {ap->v[SWAP ? 1 - stmp : stmp], ap->v[SWAP ? stmp : 1 - stmp]};
    alignas(16) std::int32_t px[MAX_HL + 8], py[MAX_HL + 8], nz[MAX_HL + 8];
    int nnz = 0;
    int32x4_t idx = {0, 1, 2, 3};
    const int32x4_t four = vdupq_n_s32(4);
    const uint32x4_t bit = {1u, 2u, 4u, 8u};
    for (int blk = 0; blk < 2; ++blk) {
        const std::int32_t* const v = blocks[blk];
        for (int i = 0; i < half; i += 4) {
            const int32x4_t x = vld1q_s32(v + i), y = vld1q_s32(v + i + half);
            const unsigned code = vaddvq_u32(vandq_u32(vandq_u32(vcgtq_s32(x, iz), vcgtq_s32(y, iz)), bit));
            const uint8x16_t sh = vld1q_u8(kFxnPack[code].data());
            vst1q_s32(px + nnz, vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(x), sh)));
            vst1q_s32(py + nnz, vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(y), sh)));
            vst1q_s32(nz + nnz, vreinterpretq_s32_u8(vqtbl1q_u8(vreinterpretq_u8_s32(idx), sh)));
            nnz += kNnzPop[code];
            idx = vaddq_s32(idx, four);
        }
    }
    for (int k = 0; k < nnz; k += 4) {
        // crelu(a) * crelu(b) of (float)v * 2^-22 (the §L4 fill and §2.3 pairwise, lane for lane)
        const float32x4_t p = vmulq_f32(
            vminq_f32(vmaxq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(px + k)), inv), vz), vo),
            vminq_f32(vmaxq_f32(vmulq_f32(vcvtq_f32_s32(vld1q_s32(py + k)), inv), vz), vo));
        const int n4 = std::min(nnz - k, 4);
        alignas(16) float hf[4];
        vst1q_f32(hf, p);
        for (int j = 0; j < n4; ++j) {
            const float hi = hf[j];
            const float* w = wbase + static_cast<std::size_t>(nz[k + j]) * istride;
            a0=vfmaq_n_f32(a0, vld1q_f32(w),    hi); a1=vfmaq_n_f32(a1, vld1q_f32(w+4),  hi);
            a2=vfmaq_n_f32(a2, vld1q_f32(w+8),  hi); a3=vfmaq_n_f32(a3, vld1q_f32(w+12), hi);
            a4=vfmaq_n_f32(a4, vld1q_f32(w+16), hi); a5=vfmaq_n_f32(a5, vld1q_f32(w+20), hi);
            a6=vfmaq_n_f32(a6, vld1q_f32(w+24), hi); a7=vfmaq_n_f32(a7, vld1q_f32(w+28), hi);
        }
    }
    alignas(16) float x1[MAX_L2];
    const float32x4_t acc[8] = {a0, a1, a2, a3, a4, a5, a6, a7};
    for (int q = 0; q < 8; ++q) {   // screlu: same ops per lane (clamp, square)
        const float32x4_t c = vminq_f32(vmaxq_f32(acc[q], vz), vo);
        vst1q_f32(x1 + 4 * q, vmulq_f32(c, c));
    }
    const Value val = finish_body(n, x1, b);
    return SWAP ? -val : val;
}
#endif

#if SC_FXP || SC_FXN
// The float eval before FXP/FXN: its fallback (dense-L1 probe, L2 != 32, HL % 64 != 0).
#define SC_FX_EVAL0 eval_float_t0
#else
#define SC_FX_EVAL0 eval_float_t
#endif

template <bool SWAP>
[[gnu::noinline]] Value SC_FX_EVAL0(Context& cx, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    // §3.1: read the incremental fixed-point accumulator (int32, scale FX_S) and
    // convert to float. Integer add is associative, so the incremental value is
    // bit-exact to a from-scratch refresh_into_fx; both produce the same integer-cp
    // eval as the old float gather (verified). Mirrors eval_quant's structure.
    alignas(64) float acc_stm[MAX_HL], acc_ntm[MAX_HL];
    const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
    auto fill = [&](const AccFx& a) {
#if defined(__ARM_NEON)
        // §L4: vectorised int32->float. Byte-identical after CReLU: a value lands in
        // (0,1) only for v in [0, FX_S=2^22), where (float)v is exact (v < 2^24) and
        // x2^-FX_SHIFT is an exact power-of-two scale; all other v are CReLU-clamped
        // to 0/1 identically regardless of the int->float rounding point.
        const float32x4_t inv = vdupq_n_f32(float(1.0 / FX_S));
        for (int j = 0; j < hl; j += 4) {
            vst1q_f32(acc_stm + j, vmulq_f32(vcvtq_f32_s32(vld1q_s32(&a.v[stmp][j])),     inv));
            vst1q_f32(acc_ntm + j, vmulq_f32(vcvtq_f32_s32(vld1q_s32(&a.v[1 - stmp][j])), inv));
        }
#else
#if defined(__AVX2__)
        if constexpr (SC_FX_AVX2) {   // the NEON §L4 form: (float)v * 2^-22, equal to the scalar one after CReLU
            const __m256 inv = _mm256_set1_ps(float(1.0 / FX_S));
            for (int j = 0; j < hl; j += 8) {
                _mm256_store_ps(acc_stm + j, _mm256_mul_ps(_mm256_cvtepi32_ps(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a.v[stmp][j]))), inv));
                _mm256_store_ps(acc_ntm + j, _mm256_mul_ps(_mm256_cvtepi32_ps(
                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a.v[1 - stmp][j]))), inv));
            }
            return;
        }
#endif
        for (int j = 0; j < hl; ++j) {
            acc_stm[j] = float(double(a.v[stmp][j])     / FX_S);
            acc_ntm[j] = float(double(a.v[1 - stmp][j]) / FX_S);
        }
#endif
    };
    if (cx.ply >= 0) {  // search has the incremental accumulator in sync
        if constexpr (SC_FX_LAZY) fx_ensure(cx, cx.ply);
        const AccFx& a = cx.stack_fx[cx.ply];
        if (g_verify) {  // incremental must equal a fresh refresh, bit-exact
            AccFx chk; refresh_into_fx(chk, board);
            for (int j = 0; j < hl; ++j)
                if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
                    std::fprintf(stderr, "[nnue] FX ACC MISMATCH ply=%d j=%d inc=%d,%d chk=%d,%d\n",
                                 cx.ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
                    break;
                }
        }
        fill(a);
    } else {  // outside search (bare `eval`): from-scratch fixed-point
        AccFx a; refresh_into_fx(a, board); fill(a);
    }

    alignas(64) float h[MAX_HL];
    // Pairwise destinations (see eval_quant_t): SWAP exchanges the two perspectives, i.e. the
    // eval of the position with the side to move flipped (the position after a pass).
    const int off_stm = SWAP ? half : 0;
    const int off_ntm = SWAP ? 0 : half;
#if defined(__ARM_NEON)
    // §2.3: elementwise crelu(a)*crelu(b), 4 lanes/iter. clamp = vmin(vmax(x,0),1),
    // bit-identical to std::clamp for finite accumulator values. half % 4 == 0.
    const float32x4_t vz = vdupq_n_f32(0.0f), vo = vdupq_n_f32(1.0f);
    for (int i = 0; i < half; i += 4) {
        const float32x4_t a = vminq_f32(vmaxq_f32(vld1q_f32(acc_stm + i),        vz), vo);
        const float32x4_t b = vminq_f32(vmaxq_f32(vld1q_f32(acc_stm + i + half), vz), vo);
        vst1q_f32(h + off_stm + i, vmulq_f32(a, b));
        const float32x4_t c = vminq_f32(vmaxq_f32(vld1q_f32(acc_ntm + i),        vz), vo);
        const float32x4_t d = vminq_f32(vmaxq_f32(vld1q_f32(acc_ntm + i + half), vz), vo);
        vst1q_f32(h + off_ntm + i, vmulq_f32(c, d));
    }
#else
    int pw0 = 0;
#if defined(__AVX2__)
    if constexpr (SC_FX_AVX2) {   // the NEON §2.3 form, 8 lanes (no input is -0.0: an int 0 converts to +0)
        const __m256 vz = _mm256_setzero_ps(), vo = _mm256_set1_ps(1.0f);
        auto cl = [&](const float* q) { return _mm256_min_ps(_mm256_max_ps(_mm256_load_ps(q), vz), vo); };
        for (; pw0 + 8 <= half; pw0 += 8) {
            _mm256_storeu_ps(h + off_stm + pw0, _mm256_mul_ps(cl(acc_stm + pw0), cl(acc_stm + pw0 + half)));
            _mm256_storeu_ps(h + off_ntm + pw0, _mm256_mul_ps(cl(acc_ntm + pw0), cl(acc_ntm + pw0 + half)));
        }
    }
#endif
    for (int i = pw0; i < half; ++i) {
        h[off_stm + i] = crelu(acc_stm[i]) * crelu(acc_stm[i + half]);
        h[off_ntm + i] = crelu(acc_ntm[i]) * crelu(acc_ntm[i + half]);
    }
#endif
    const int b = n.obucket[board.occ().count()];
    float x1[MAX_L2];
#if defined(__ARM_NEON)
    // Row i at wbase + i*istride + o: the bucket-major copy when built (FXB), else the [i][bucket][o] rows.
    const bool bm = !n.l1w_bm.empty();
    const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
    const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                            : &n.l1w[static_cast<std::size_t>(b) * L2];
    const float* l1bb  = &n.l1b[b * L2];
    if (!g_l1dense && L2 == 32) {
        // §L1 NNZ sparse (input-major): iterate only nonzero h[i]. A skipped h[i]==0 would
        // add w*0.0f == 0 exactly, so per output the ascending-i fmla chain is unchanged ->
        // BYTE-IDENTICAL to the dense §2.4 path. ~76% of h is zero (round-1 h8 measurement).
        // 8 f32x4 accumulators hold all 32 outputs; the 32 o-weights at a fixed i are contiguous.
        int nz[MAX_HL];
        const int nnz = build_nnz(h, hl, nz);  // §P1: NEON left-packing (same list, faster)
        float32x4_t a0=vld1q_f32(l1bb),    a1=vld1q_f32(l1bb+4),  a2=vld1q_f32(l1bb+8),
                    a3=vld1q_f32(l1bb+12), a4=vld1q_f32(l1bb+16), a5=vld1q_f32(l1bb+20),
                    a6=vld1q_f32(l1bb+24), a7=vld1q_f32(l1bb+28);
        for (int k = 0; k < nnz; ++k) {
            const int i = nz[k];
            const float hi = h[i];
            const float* w = wbase + static_cast<std::size_t>(i) * istride;
            a0=vfmaq_n_f32(a0, vld1q_f32(w),    hi); a1=vfmaq_n_f32(a1, vld1q_f32(w+4),  hi);
            a2=vfmaq_n_f32(a2, vld1q_f32(w+8),  hi); a3=vfmaq_n_f32(a3, vld1q_f32(w+12), hi);
            a4=vfmaq_n_f32(a4, vld1q_f32(w+16), hi); a5=vfmaq_n_f32(a5, vld1q_f32(w+20), hi);
            a6=vfmaq_n_f32(a6, vld1q_f32(w+24), hi); a7=vfmaq_n_f32(a7, vld1q_f32(w+28), hi);
        }
        alignas(16) float tmp[32];
        vst1q_f32(tmp,a0);    vst1q_f32(tmp+4,a1);  vst1q_f32(tmp+8,a2);  vst1q_f32(tmp+12,a3);
        vst1q_f32(tmp+16,a4); vst1q_f32(tmp+20,a5); vst1q_f32(tmp+24,a6); vst1q_f32(tmp+28,a7);
        for (int o = 0; o < 32; ++o) x1[o] = screlu(tmp[o]);
    } else {
        // dense §2.4 (SCNNUE_L1DENSE A/B baseline, or L2 != 32)
        for (int o = 0; o < L2; o += 4) {
            float32x4_t acc = vld1q_f32(l1bb + o);
            for (int i = 0; i < hl; ++i)
                acc = vfmaq_n_f32(acc, vld1q_f32(wbase + static_cast<std::size_t>(i) * istride + o), h[i]);
            float tmp[4];
            vst1q_f32(tmp, acc);
            for (int k = 0; k < 4; ++k) x1[o + k] = screlu(tmp[k]);
        }
    }
#else
#if defined(__AVX2__)
    if (SC_FX_AVX2 && !g_l1dense && L2 == 32 && (hl & 63) == 0) {
        // The NEON §L1 form: input-major over the nonzero h[i] (ascending: a 64-bit mask per 64 inputs,
        // walked with tzcnt), all 32 outputs in 4 ymm. Per output that is the scalar loop's fma chain
        // (s = fma(h[i], w, s), ascending i, from the bias) minus terms that add +0 -- the same value,
        // a +-0 sum only flipping a zero's sign, which screlu erases.
        const bool bm = !n.l1w_bm.empty();
        const std::size_t istride = bm ? 32 : static_cast<std::size_t>(OB) * L2;
        const float* wbase = bm ? n.l1w_bm.data() + static_cast<std::size_t>(b) * hl * 32
                                : &n.l1w[static_cast<std::size_t>(b) * L2];
        const float* l1bb  = &n.l1b[b * L2];
        __m256 a0 = _mm256_loadu_ps(l1bb), a1 = _mm256_loadu_ps(l1bb + 8),
               a2 = _mm256_loadu_ps(l1bb + 16), a3 = _mm256_loadu_ps(l1bb + 24);
        const __m256 vz = _mm256_setzero_ps();
        for (int g = 0; g < hl; g += 64) {
            std::uint64_t m = 0;
            for (int t = 0; t < 8; ++t)
                m |= static_cast<std::uint64_t>(static_cast<unsigned>(_mm256_movemask_ps(
                         _mm256_cmp_ps(_mm256_load_ps(h + g + 8 * t), vz, _CMP_NEQ_OQ)))) << (8 * t);
            while (m) {
                const int i = g + __builtin_ctzll(m);
                m &= m - 1;
                const __m256 hv = _mm256_set1_ps(h[i]);
                const float* w = wbase + static_cast<std::size_t>(i) * istride;
                a0 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w),      a0);
                a1 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 8),  a1);
                a2 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 16), a2);
                a3 = _mm256_fmadd_ps(hv, _mm256_loadu_ps(w + 24), a3);
            }
        }
        const __m256 acc[4] = {a0, a1, a2, a3};
        for (int q = 0; q < 4; ++q) {   // 8-wide screlu, same ops per lane (clamp, square)
            const __m256 v = _mm256_min_ps(_mm256_max_ps(acc[q], vz), _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(x1 + 8 * q, _mm256_mul_ps(v, v));
        }
    } else
#endif
    for (int o = 0; o < L2; ++o) {
        float s = n.l1b[b * L2 + o];
        for (int i = 0; i < hl; ++i)
            s += h[i] * n.l1w[static_cast<std::size_t>(i) * (OB * L2) + b * L2 + o];
        x1[o] = screlu(s);
    }
#endif
    const Value v = finish_body(n, x1, b);
    return SWAP ? -v : v;
}

// Quantised (SCN5): int8 feature transformer (x QA=127) with int16 bias and int16
// accumulator; crelu clamps to [0,QA], the pairwise product of the two halves is
// scaled back to int8 [0,QA] (h8), which feeds the int8 L1 sdot; the L1 output is
// dequantised (by QA*QB) to float; body kept float.
//
// Speed path: fixed stack buffers (no per-eval heap alloc) and an int16 accumulator
// so the hot add loop widens the int8 weights 16-wide on NEON (vmovl_s8 + vaddq_s16).
// int16 headroom: |l0w_i8| <= QA=127 (l0w is clipped to +/-0.99 in training, then
// x127), so the accumulator holds ~250 active features before nearing 32767; real
// positions peak ~112 (verified over 200k positions), so it does not overflow.
// SWAP as in eval_float_t: false = the position as it is, true = the perspectives exchanged
// (the position after a pass), negated to the original side to move's point of view.
// SCNNUE_VERIFY: the incremental accumulator against a fresh recompute (its 2 KB Acc stays out
// of evaluate()'s frame).
[[gnu::noinline, gnu::cold]] void verify_acc_now(const Acc& a, int stmp, const Board& board, int ply) {
    const int hl = g_net.hl;
    Acc chk;
    refresh_into(chk, board);
    for (int j = 0; j < hl; ++j)
        if (a.v[stmp][j] != chk.v[stmp][j] || a.v[1 - stmp][j] != chk.v[1 - stmp][j]) {
            std::fprintf(stderr, "[nnue] ACC MISMATCH ply=%d j=%d  inc=%d,%d chk=%d,%d\n",
                         ply, j, a.v[stmp][j], a.v[1 - stmp][j], chk.v[stmp][j], chk.v[1 - stmp][j]);
            break;
        }
}
template <bool SWAP>
Value eval_quant_t(Context& cx, const Board& board) {
    const Net& n = g_net;
    const int hl = n.hl, half = hl / 2, L2 = n.l2;

    // The incremental path reads the accumulator in place through these pointers (no 2KB
    // memcpy — that copy was profiled at ~11% of eval on the play path; byte-identical, same
    // int16 values). The gather fallback fills the Context's buffers.
    const std::int16_t* acc_stm;
    const std::int16_t* acc_ntm;
    const std::int8_t* l0 = n.l0w_i8.data();

    // Full accumulator (base + threats + pp) from the incremental state if the
    // search has it in sync — no gather at all. Else full recompute (fallback).
    const bool have_acc = (cx.ply >= 0);
    if (have_acc) {
        const int stmp = (board.sideToMove() == Color::WHITE) ? 0 : 1;
#if SC_Q_LAZY
        q_ensure(cx, cx.ply);
#endif
        const Acc& a = cx.stack[cx.ply];
        acc_stm = a.v[stmp];       // read the accumulator in place — no copy
        acc_ntm = a.v[1 - stmp];
        if (g_verify) verify_acc_now(a, stmp, board, cx.ply);   // vs a fresh recompute
    } else {
        std::int16_t* acc_buf_stm = cx.fb[0];
        std::int16_t* acc_buf_ntm = cx.fb[1];
        for (int j = 0; j < hl; ++j) { acc_buf_stm[j] = n.l0b_i[j]; acc_buf_ntm[j] = n.l0b_i[j]; }
        gather(
            board, n.base_dims,
            [&](std::size_t f) { acc_add(acc_buf_stm, l0 + f * hl, hl); },
            [&](std::size_t f) { acc_add(acc_buf_ntm, l0 + f * hl, hl); });
        acc_stm = acc_buf_stm;
        acc_ntm = acc_buf_ntm;
    }

    // Int8 pairwise activation: crelu-clamp to [0,QA], multiply the
    // two halves, and scale down by QA so the result fits int8 [0,QA]. This is
    // the L1 input for the sdot dot-products (h8 = pairwise * QA).
    alignas(16) std::int8_t h8[MAX_HL];
    // Where each perspective's pairwise products land in h8. SWAP=false: side to move first
    // (the position as it is). SWAP=true: the two perspectives exchanged, i.e. the net sees the
    // position with the side to move flipped -- the position after a pass. Nothing above this
    // line depends on the side to move (Acc::v is per absolute colour, the output bucket is
    // occupancy-only), so the swapped query is exactly the eval of the null-moved position.
    const int off_stm = SWAP ? half : 0;
    const int off_ntm = SWAP ? 0 : half;
#if defined(__ARM_NEON)
    // NEON, 16 lanes/iter (P2). Bit-identical to the scalar form: clamped products are
    // <= QA*QA = 16129 (fit u16), and floor(x/127) == (x + 1 + ((x + 1) >> 7)) >> 7 exactly for
    // x in [0,16129] (checked for every x): vmla (1 + a*b), vsra (+ >>7), vshrn (>>7, to u8).
    static_assert(QA == 127, "NEON pairwise /QA magic assumes QA==127");
    const int16x8_t vz = vdupq_n_s16(0), vqa = vdupq_n_s16(QA);
    {
        const uint16x8_t one = vdupq_n_u16(1);
        auto pw16 = [&](const std::int16_t* src, std::int8_t* dst) {
            const uint16x8_t a0 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src), vz), vqa));
            const uint16x8_t a1 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src + 8), vz), vqa));
            const uint16x8_t b0 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src + half), vz), vqa));
            const uint16x8_t b1 = vreinterpretq_u16_s16(vminq_s16(vmaxq_s16(vld1q_s16(src + half + 8), vz), vqa));
            uint16x8_t x0 = vmlaq_u16(one, a0, b0), x1 = vmlaq_u16(one, a1, b1);   // 1 + a*b
            x0 = vsraq_n_u16(x0, x0, 7);  x1 = vsraq_n_u16(x1, x1, 7);             // + (1 + a*b) >> 7
            vst1q_u8(reinterpret_cast<std::uint8_t*>(dst), vshrn_high_n_u16(vshrn_n_u16(x0, 7), x1, 7));
        };
        for (int i = 0; i < half; i += 16) {
            pw16(acc_stm + i, h8 + off_stm + i);
            pw16(acc_ntm + i, h8 + off_ntm + i);
        }
    }
#else
    auto cr = [](int x) -> int { return x < 0 ? 0 : (x > QA ? QA : x); };
    for (int i = 0; i < half; ++i) {
        h8[off_stm + i] = static_cast<std::int8_t>((cr(acc_stm[i]) * cr(acc_stm[i + half])) / QA);
        h8[off_ntm + i] = static_cast<std::int8_t>((cr(acc_ntm[i]) * cr(acc_ntm[i + half])) / QA);
    }
#endif
    const int b = n.obucket[board.occ().count()];
    const float DEQ = float(QA) * QB;  // h8 already carries the 1/QA scale
    const std::int8_t* wbucket = n.l1w_dot.data() + static_cast<std::size_t>(b) * L2 * hl;
    float x1[MAX_L2];
#if defined(__ARM_FEATURE_DOTPROD)
    if (L2 == 32 && !n.l1w_sp.empty()) {
        // S1 sparse L1: list the nonzero 4-byte chunks of h8 (ascending), then for each one add its
        // 4 inputs x all 32 outputs (eight vdotq over one 128-byte weight block). Two accumulator
        // sets alternate chunks to halve the dependency chains. Zero chunks add exactly 0, and
        // integer sums are order-independent -> the same int32 per output as the dense form.
        alignas(16) std::uint16_t nz[MAX_HL / 4 + 8];
        int nnz = 0;
        const std::uint32_t* h32 = reinterpret_cast<const std::uint32_t*>(h8);
        const uint32x4_t bit = {1u, 2u, 4u, 8u};
        for (int k = 0; k < hl / 4; k += 4) {
            const uint32x4_t v = vld1q_u32(h32 + k);
            const unsigned m = vaddvq_u32(vandq_u32(vtstq_u32(v, v), bit));
            vst1_u16(nz + nnz, vadd_u16(vdup_n_u16(static_cast<std::uint16_t>(k)), vld1_u16(kNnzLut16[m])));
            nnz += kNnzPop[m];
        }
        const std::int8_t* W = n.l1w_sp.data() + static_cast<std::size_t>(b) * (hl / 4) * 128;
        int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        int32x4_t c0 = a0, c1 = a0, c2 = a0, c3 = a0, c4 = a0, c5 = a0, c6 = a0, c7 = a0;
        int j = 0;
        for (; j + 2 <= nnz; j += 2) {
            const int k0 = nz[j], k1 = nz[j + 1];
            const int8x16_t i0 = vreinterpretq_s8_u32(vdupq_n_u32(h32[k0]));
            const int8x16_t i1 = vreinterpretq_s8_u32(vdupq_n_u32(h32[k1]));
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            const std::int8_t* w1 = W + static_cast<std::size_t>(k1) * 128;
            a0 = vdotq_s32(a0, vld1q_s8(w0),      i0);  c0 = vdotq_s32(c0, vld1q_s8(w1),      i1);
            a1 = vdotq_s32(a1, vld1q_s8(w0 + 16), i0);  c1 = vdotq_s32(c1, vld1q_s8(w1 + 16), i1);
            a2 = vdotq_s32(a2, vld1q_s8(w0 + 32), i0);  c2 = vdotq_s32(c2, vld1q_s8(w1 + 32), i1);
            a3 = vdotq_s32(a3, vld1q_s8(w0 + 48), i0);  c3 = vdotq_s32(c3, vld1q_s8(w1 + 48), i1);
            a4 = vdotq_s32(a4, vld1q_s8(w0 + 64), i0);  c4 = vdotq_s32(c4, vld1q_s8(w1 + 64), i1);
            a5 = vdotq_s32(a5, vld1q_s8(w0 + 80), i0);  c5 = vdotq_s32(c5, vld1q_s8(w1 + 80), i1);
            a6 = vdotq_s32(a6, vld1q_s8(w0 + 96), i0);  c6 = vdotq_s32(c6, vld1q_s8(w1 + 96), i1);
            a7 = vdotq_s32(a7, vld1q_s8(w0 + 112), i0); c7 = vdotq_s32(c7, vld1q_s8(w1 + 112), i1);
        }
        if (j < nnz) {
            const int k0 = nz[j];
            const int8x16_t i0 = vreinterpretq_s8_u32(vdupq_n_u32(h32[k0]));
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            a0 = vdotq_s32(a0, vld1q_s8(w0),      i0);  a1 = vdotq_s32(a1, vld1q_s8(w0 + 16), i0);
            a2 = vdotq_s32(a2, vld1q_s8(w0 + 32), i0);  a3 = vdotq_s32(a3, vld1q_s8(w0 + 48), i0);
            a4 = vdotq_s32(a4, vld1q_s8(w0 + 64), i0);  a5 = vdotq_s32(a5, vld1q_s8(w0 + 80), i0);
            a6 = vdotq_s32(a6, vld1q_s8(w0 + 96), i0);  a7 = vdotq_s32(a7, vld1q_s8(w0 + 112), i0);
        }
        alignas(16) std::int32_t sum[32];
        vst1q_s32(sum,      vaddq_s32(a0, c0));  vst1q_s32(sum + 4,  vaddq_s32(a1, c1));
        vst1q_s32(sum + 8,  vaddq_s32(a2, c2));  vst1q_s32(sum + 12, vaddq_s32(a3, c3));
        vst1q_s32(sum + 16, vaddq_s32(a4, c4));  vst1q_s32(sum + 20, vaddq_s32(a5, c5));
        vst1q_s32(sum + 24, vaddq_s32(a6, c6));  vst1q_s32(sum + 28, vaddq_s32(a7, c7));
        const float* bias = n.l1b.data() + b * L2;
        for (int o = 0; o < 32; ++o) x1[o] = screlu(static_cast<float>(sum[o]) / DEQ + bias[o]);
    } else
    // NEON campaign N5 (A1-analog): 4 outputs per pass, each on TWO independent vdotq chains --
    // 8 dot chains in flight instead of one serial 32-op chain per output (latency-bound), and
    // every h8 vector is loaded once for 4 outputs. Integer sums are order-independent and
    // cannot overflow (|sum| <= 512*127*127) -> bit-identical to the one-chain form.
    {
        int o = 0;
        for (; o + 4 <= L2; o += 4) {
            const std::int8_t* w0 = wbucket + static_cast<std::size_t>(o) * hl;
            const std::int8_t* w1 = w0 + hl;
            const std::int8_t* w2 = w1 + hl;
            const std::int8_t* w3 = w2 + hl;
            int32x4_t a0 = vdupq_n_s32(0), a1 = a0, a2 = a0, a3 = a0;
            int32x4_t c0 = a0, c1 = a0, c2 = a0, c3 = a0;
            int i = 0;
            for (; i + 32 <= hl; i += 32) {
                const int8x16_t h = vld1q_s8(h8 + i), h2 = vld1q_s8(h8 + i + 16);
                a0 = vdotq_s32(a0, h, vld1q_s8(w0 + i));  c0 = vdotq_s32(c0, h2, vld1q_s8(w0 + i + 16));
                a1 = vdotq_s32(a1, h, vld1q_s8(w1 + i));  c1 = vdotq_s32(c1, h2, vld1q_s8(w1 + i + 16));
                a2 = vdotq_s32(a2, h, vld1q_s8(w2 + i));  c2 = vdotq_s32(c2, h2, vld1q_s8(w2 + i + 16));
                a3 = vdotq_s32(a3, h, vld1q_s8(w3 + i));  c3 = vdotq_s32(c3, h2, vld1q_s8(w3 + i + 16));
            }
            for (; i < hl; i += 16) {   // hl % 32 tail (empty for hl = 512)
                const int8x16_t h = vld1q_s8(h8 + i);
                a0 = vdotq_s32(a0, h, vld1q_s8(w0 + i));  a1 = vdotq_s32(a1, h, vld1q_s8(w1 + i));
                a2 = vdotq_s32(a2, h, vld1q_s8(w2 + i));  a3 = vdotq_s32(a3, h, vld1q_s8(w3 + i));
            }
            const float* bias = n.l1b.data() + b * L2 + o;
            x1[o]     = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a0, c0))) / DEQ + bias[0]);
            x1[o + 1] = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a1, c1))) / DEQ + bias[1]);
            x1[o + 2] = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a2, c2))) / DEQ + bias[2]);
            x1[o + 3] = screlu(static_cast<float>(vaddvq_s32(vaddq_s32(a3, c3))) / DEQ + bias[3]);
        }
        for (; o < L2; ++o) {   // L2 % 4 tail (empty for L2 = 32)
            const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
            int32x4_t acc4 = vdupq_n_s32(0);
            for (int i = 0; i < hl; i += 16) acc4 = vdotq_s32(acc4, vld1q_s8(h8 + i), vld1q_s8(w + i));
            x1[o] = screlu(static_cast<float>(vaddvq_s32(acc4)) / DEQ + n.l1b[b * L2 + o]);
        }
    }
#elif defined(__AVX2__)
    if (L2 == 32 && (hl & 31) == 0 && !n.l1w_sp.empty()) {
        // S1 sparse L1 on AVX2: list the nonzero 4-byte chunks of h8 (8 per compare, one LUT store,
        // no branch), then per chunk broadcast its 4 inputs and add them x all 32 outputs: maddubs
        // (u8 h8 <= 127 x s8 weight; a pair sum is at most 2*127*128 = 32512, no int16 saturation)
        // + madd(., 1) -> 8 int32 outputs per vector, 4 vectors. Two accumulator sets alternate
        // chunks. Zero chunks add exactly 0 and int32 sums are order-independent, so every output is
        // the dense kernel's integer; the epilogue is the same IEEE ops per lane.
        alignas(32) std::uint16_t nz[MAX_HL / 4 + 8];
        int nnz = 0;
        const __m256i zero = _mm256_setzero_si256();
        __m128i base = _mm_setzero_si128();
        const __m128i eight = _mm_set1_epi16(8);
        for (int i = 0; i < hl; i += 32) {
            const __m256i h = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(h8 + i));
            const unsigned m = ~static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(h, zero)))) & 0xFFu;
            _mm_storeu_si128(reinterpret_cast<__m128i*>(nz + nnz),
                             _mm_add_epi16(base, _mm_load_si128(reinterpret_cast<const __m128i*>(kNnzLut8[m].data()))));
            nnz += __builtin_popcount(m);
            base = _mm_add_epi16(base, eight);
        }
        const std::int8_t* W = n.l1w_sp.data() + static_cast<std::size_t>(b) * (hl / 4) * 128;
        const __m256i ones16 = _mm256_set1_epi16(1);
        auto in4 = [&](int k) { std::int32_t v; std::memcpy(&v, h8 + 4 * k, 4); return _mm256_set1_epi32(v); };
        auto wv = [](const std::int8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        auto dp = [&](__m256i acc, __m256i in, const std::int8_t* w) {
            return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(in, wv(w)), ones16));
        };
        __m256i a0 = zero, a1 = zero, a2 = zero, a3 = zero, c0 = zero, c1 = zero, c2 = zero, c3 = zero;
        int j = 0;
        for (; j + 2 <= nnz; j += 2) {
            const int k0 = nz[j], k1 = nz[j + 1];
            const __m256i i0 = in4(k0), i1 = in4(k1);
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            const std::int8_t* w1 = W + static_cast<std::size_t>(k1) * 128;
            a0 = dp(a0, i0, w0);      c0 = dp(c0, i1, w1);
            a1 = dp(a1, i0, w0 + 32); c1 = dp(c1, i1, w1 + 32);
            a2 = dp(a2, i0, w0 + 64); c2 = dp(c2, i1, w1 + 64);
            a3 = dp(a3, i0, w0 + 96); c3 = dp(c3, i1, w1 + 96);
        }
        if (j < nnz) {
            const int k0 = nz[j];
            const __m256i i0 = in4(k0);
            const std::int8_t* w0 = W + static_cast<std::size_t>(k0) * 128;
            a0 = dp(a0, i0, w0);      a1 = dp(a1, i0, w0 + 32);
            a2 = dp(a2, i0, w0 + 64); a3 = dp(a3, i0, w0 + 96);
        }
        const __m256i s[4] = {_mm256_add_epi32(a0, c0), _mm256_add_epi32(a1, c1),
                              _mm256_add_epi32(a2, c2), _mm256_add_epi32(a3, c3)};
        const float* bias = n.l1b.data() + b * L2;
        for (int q = 0; q < 4; ++q) {   // the dense kernel's epilogue (A7), 8 lanes at a time
            __m256 v = _mm256_add_ps(_mm256_div_ps(_mm256_cvtepi32_ps(s[q]), _mm256_set1_ps(DEQ)),
                                     _mm256_loadu_ps(bias + 8 * q));
            v = _mm256_min_ps(_mm256_max_ps(v, _mm256_setzero_ps()), _mm256_set1_ps(1.0f));
            _mm256_storeu_ps(x1 + 8 * q, _mm256_mul_ps(v, v));
        }
    } else
    // AVX2 counterpart of the sdot path. h8 is in [0,QA]=[0,127], i.e. a valid UNSIGNED
    // byte, and l1w_dot is signed int8: VPMADDUBSW (u8*s8, adjacent pairs summed into
    // int16) then VPMADDWD (int16 pairs -> int32). Saturation is impossible: a pair sum
    // is at most 2*127*127 = 32258 < 32767. Integer sums are order-independent, so the
    // result equals the scalar int32 accumulation exactly (bit-identical). Four outputs
    // share every 32-byte h8 load; the epilogue is the scalar one, verbatim.
    {
        const __m256i ones16 = _mm256_set1_epi16(1);
        auto hsum32 = [](__m256i v) -> std::int32_t {
            __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));
            s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1));
            return _mm_cvtsi128_si32(s);
        };
        auto ld = [](const std::int8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); };
        int o = 0;
        for (; o + 4 <= L2; o += 4) {
            const std::int8_t* w0 = wbucket + static_cast<std::size_t>(o) * hl;
            const std::int8_t* w1 = w0 + hl;
            const std::int8_t* w2 = w1 + hl;
            const std::int8_t* w3 = w2 + hl;
            __m256i a0 = _mm256_setzero_si256(), a1 = a0, a2 = a0, a3 = a0;
            int i = 0;
            for (; i + 32 <= hl; i += 32) {
                const __m256i h = ld(h8 + i);
                a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w0 + i)), ones16));
                a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w1 + i)), ones16));
                a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w2 + i)), ones16));
                a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(h, ld(w3 + i)), ones16));
            }
            std::int32_t s0 = hsum32(a0), s1 = hsum32(a1), s2 = hsum32(a2), s3 = hsum32(a3);
            for (; i < hl; ++i) {  // hl % 32 tail (empty for hl = 512)
                s0 += static_cast<int>(h8[i]) * w0[i]; s1 += static_cast<int>(h8[i]) * w1[i];
                s2 += static_cast<int>(h8[i]) * w2[i]; s3 += static_cast<int>(h8[i]) * w3[i];
            }
            {   // 4-wide epilogue -- the same IEEE ops per lane as the scalar form
                // (int->float, divide by DEQ, add bias, clamp to [0,1], square). x86 campaign A7: +1.04%.
                const __m128 sv = _mm_cvtepi32_ps(_mm_setr_epi32(s0, s1, s2, s3));
                __m128 v = _mm_add_ps(_mm_div_ps(sv, _mm_set1_ps(DEQ)), _mm_loadu_ps(n.l1b.data() + b * L2 + o));
                v = _mm_min_ps(_mm_max_ps(v, _mm_setzero_ps()), _mm_set1_ps(1.0f));
                _mm_storeu_ps(x1 + o, _mm_mul_ps(v, v));
            }
        }
        for (; o < L2; ++o) {  // L2 % 4 tail (empty for L2 = 32)
            const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
            std::int32_t s = 0;
            for (int i = 0; i < hl; ++i) s += static_cast<int>(h8[i]) * w[i];
            x1[o] = screlu(static_cast<float>(s) / DEQ + n.l1b[b * L2 + o]);
        }
    }
#else
    for (int o = 0; o < L2; ++o) {
        const std::int8_t* w = wbucket + static_cast<std::size_t>(o) * hl;
        std::int32_t s = 0;
        for (int i = 0; i < hl; ++i) s += static_cast<int>(h8[i]) * w[i];
        x1[o] = screlu(static_cast<float>(s) / DEQ + n.l1b[b * L2 + o]);
    }
#endif
    const Value v = finish_body(n, x1, b);
    return SWAP ? -v : v;
}

}  // namespace

Value evaluate(Context& cx, const Board& board) {
    // EVC: the raw eval is a pure function of the position, and the search can ask for the same one again
    // (a node that returned before its TT store), as can another thread. A hit skips the eval and, on the
    // float path while the ply's accumulator is unread, its column pass too. The search sees the same value
    // either way. An entry: the key's top 48 bits and the eval (|eval| <= 15000, finish_body), one atomic
    // word, so no lock and no torn read; with the index bits (at least 16) it compares the whole key.
    const std::uint64_t key = board.hash();
    std::uint64_t* const slot = &g_evc[key & g_evc_mask];
    const std::uint64_t e = __atomic_load_n(slot, __ATOMIC_RELAXED);
    if ((e >> 16) == (key >> 16)) return static_cast<Value>(static_cast<std::int16_t>(e & 0xFFFF));
    const Value v = g_net.quant ? eval_quant_t<false>(cx, board) : eval_float_t<false>(cx, board);
    __atomic_store_n(slot, (key & ~std::uint64_t(0xFFFF)) | static_cast<std::uint16_t>(v), __ATOMIC_RELAXED);
    return v;
}

void prefetch_eval(std::uint64_t key) { __builtin_prefetch(&g_evc[key & g_evc_mask], 1, 3); }

void eval_cache_resize(int threads) {
    int bits = SC_EVC_BITS;
    while ((1 << (bits - SC_EVC_BITS)) < threads) ++bits;
    std::uint64_t* p = nullptr;
#if defined(_WIN32) && SC_EVC_LARGE
    p = static_cast<std::uint64_t*>(sc_nnue_large_alloc((std::size_t(1) << bits) * sizeof(std::uint64_t)));   // zeroed
#endif
    if (!p) p = static_cast<std::uint64_t*>(std::calloc(std::size_t(1) << bits, sizeof(std::uint64_t)));
    if (!p) return;   // keep the table there is
    if (g_evc != g_evc_boot) {
#if defined(_WIN32) && SC_EVC_LARGE
        if (!sc_nnue_large_free(g_evc))
#endif
            std::free(g_evc);
    }
    g_evc = p;
    g_evc_mask = (std::size_t(1) << bits) - 1;
}

void eval_cache_clear() { std::memset(g_evc, 0, (g_evc_mask + 1) * sizeof(std::uint64_t)); }

Value evaluate_pass(Context& cx, const Board& board) {
    return g_net.quant ? eval_quant_t<true>(cx, board) : eval_float_t<true>(cx, board);
}

// ---- incremental accumulator hooks (called by search) -----------------------
void acc_reset(Context& cx, const Board& board) {
    if (!g_net.loaded) { cx.ply = -1; return; }
    if (cx.finny_gen != g_net_gen) {   // K1: cached sums belong to the previous net
        for (auto& p : cx.finny) for (auto& f : p) for (auto& e : f) e.valid = false;
        for (auto& e : cx.finny_fx) e.valid = false;
        cx.finny_gen = g_net_gen;
    }
    if (cx.bb.empty()) cx.bb.resize(ACC_STACK);
    cx.ply = 0;
    if (g_net.quant) {
        if (cx.stack.empty()) cx.stack.resize(ACC_STACK);
        refresh_into(cx.stack[0], board);
#if SC_Q_LAZY
        cx.q_ok[0] = 1;
#endif
    } else {  // float labeler: fixed-point incremental accumulator
        if (cx.stack_fx.empty()) cx.stack_fx.resize(ACC_STACK);
        if (cx.finny_fx.empty()) cx.finny_fx.resize(2 * 2 * kKingBuckets);
        if (cx.fxd_count.size() != g_net.total_inputs) cx.fxd_count.assign(g_net.total_inputs, 0);
        // FXR: the next iteration of the same root finds its accumulator still in place (only
        // acc_make writes the stack, and always above ply 0).
        if (cx.fx_root_key != board.hash() || cx.fx_root_gen != g_net_gen) {
            refresh_into_fx(cx.stack_fx[0], board);
            cx.fx_root_key = board.hash();
            cx.fx_root_gen = g_net_gen;
        }
        if constexpr (SC_FX_LAZY) {
            if (cx.fx_ok.empty()) { cx.fx_ok.resize(ACC_STACK); cx.fx_pend.reset(new Context::FxPend[ACC_STACK]); }
            cx.fx_ok[0] = 1;
        }
    }
    cx.bb[0] = BB(board);
}

// The board after a normal move, from the board before it.
inline void bb_apply(const BB& bb0, BB& bb1, int from, int to, int mcolor, int mt0, bool cap, int cpi) {
    bb1 = bb0;  // copy parent state, then apply the move in place
    const int sf_moved = (mcolor << 3) | (mt0 + 1);
    bb1.by[sf_moved] ^= (1ull << from) | (1ull << to);
    bb1.piece_on[from] = -1;
    if (cap) bb1.by[((cpi / 6) << 3) | ((cpi % 6) + 1)] &= ~(1ull << to);
    bb1.piece_on[to] = static_cast<std::int8_t>(sf_moved);
    bb1.occ = (bb0.occ & ~(1ull << from)) | (1ull << to);
    if (mt0 == 5) { if (mcolor == 0) bb1.wk = to; else bb1.bk = to; }  // king moved
}

// The columns a normal move changes, per perspective: base (with_base: they ride in the same lists as
// the threat/pp columns, G4), threats and pawn pairs. bb0/bb1: the boards before and after the move.
template <class WT>
[[gnu::always_inline]] inline void delta_lists(const BB& bb0, const BB& bb1, int from, int to, int mcolor, int mt0,
                                               bool cap, int cpi, const WT* l0, int hl, bool with_base,
                                               const WT* (&addp)[2][1024], const WT* (&subp)[2][1024],
                                               int (&na)[2], int (&ns)[2]) {
    constexpr bool FX = (sizeof(WT) != 1);
    const int sf_moved = (mcolor << 3) | (mt0 + 1);
        const int wo = orient_tbl(bb0.wk);       // white perspective (swap 0)
        const int bo = orient_tbl(bb0.bk) ^ 56;  // black perspective (swap 8)

        const std::uint64_t excl = (1ull << from) | (1ull << to);
        // ===== Direct delta: emit only genuinely-changed columns straight to the
        // fused add/sub lists -- no full-set collect, no sort, no merge-diff.
        // Moved/captured/non-slider threat indices never cancel between before and
        // after (attacker square or attacked-type always differs), so they are
        // emitted directly; each slider emits its exact a0^a1 ray change (plus the
        // from/to identity flip). Every emitted column is a real change.
        const std::size_t bd = g_net.base_dims;
        // P1: every line of a column, issued when the column is known.
        [[maybe_unused]] auto prefetch_col = [&](const WT* col) {
            const char* c = reinterpret_cast<const char*>(col);
            const std::size_t bytes = static_cast<std::size_t>(hl) * sizeof(WT);
            if constexpr (FX) {
                for (std::size_t off = 0; off < bytes && SC_FX_PF_STRIDE; off += SC_FX_PF_STRIDE + !SC_FX_PF_STRIDE) __builtin_prefetch(c + off, 0, SC_NPS_PF_LOC);
                return;
            }
            for (std::size_t off = 0; off < bytes; off += kPfStride) __builtin_prefetch(c + off, 0, SC_NPS_PF_LOC);
        };
        if (with_base) {   // G4
            const int kabs[2] = {bb0.wk, bb0.bk};
            for (int p = 0; p < 2; ++p) {
                const WT* const ba = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, to, kabs[p])) * hl;
                const WT* const b0 = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, from, kabs[p])) * hl;
                addp[p][na[p]++] = ba;
                subp[p][ns[p]++] = b0;
                if constexpr (SC_NPS_P1 || (FX && SC_FX_PF_X86)) { prefetch_col(ba); prefetch_col(b0); }
                if (cap) {
                    const WT* const b1 = l0 + static_cast<std::size_t>(base_feat(p, cpi / 6, cpi % 6, to, kabs[p])) * hl;
                    subp[p][ns[p]++] = b1;
                    if constexpr (SC_NPS_P1 || (FX && SC_FX_PF_X86)) prefetch_col(b1);
                }
            }
        }
        auto push = [&](int p, std::uint32_t idx, bool add) {
            const WT* col = l0 + (bd + idx) * hl;
            if constexpr (SC_NPS_P1 || (FX && SC_FX_PF_X86)) prefetch_col(col);
            if (add) addp[p][na[p]++] = col; else subp[p][ns[p]++] = col;
        };
        auto emit_threat = [&](int a_sf, int a_sq, int t_sq, int attacked, bool add) {
            // Only pairs that are threat features are pushed (attacks on a king and the other
            // excluded pairs map past THREAT_DIMS).
            const std::uint32_t s = make_threat_index(a_sf, a_sq ^ wo, t_sq ^ wo, attacked);
            if (s < (std::uint32_t)THREAT_DIMS) push(0, s, add);
            const std::uint32_t nn = make_threat_index(a_sf ^ 8, a_sq ^ bo, t_sq ^ bo, attacked ^ 8);
            if (nn < (std::uint32_t)THREAT_DIMS) push(1, nn, add);
        };
        // full threat set of piece `sf` on `sq` in board `b` (mover: sub@bb0, add@bb1;
        // captured: sub@bb0).
        auto emit_full = [&](const BB& b, int sq, int sf, bool add) {
            const int pt = sf & 7;
            if (pt == 6 || pt == 0) return;
            const std::uint64_t N = b.by[2] | b.by[10], Bp = b.by[3] | b.by[11],
                                R = b.by[4] | b.by[12], Q = b.by[5] | b.by[13],
                                P = b.by[SF_W_PAWN] | b.by[SF_B_PAWN];
            std::uint64_t attacks;
            if (pt == 1) attacks = pawn_attacks(sf < 8 ? 0 : 1, sq) & (N | R);
            else { const std::uint64_t tgt = (pt == 2 || pt == 5) ? (P | N | Bp | R | Q) : (P | N | Bp | R);
                   attacks = attacks_bb(pt, sq, b.occ) & tgt; }
            while (attacks) { const int t = __builtin_ctzll(attacks); attacks &= attacks - 1;
                emit_threat(sf, sq, t, b.piece_on[t], add); }
        };
        // one non-slider attacker's threat to a single target square on a board
        auto emit_ns = [&](int a_sq, int a_sf, int tgt, const BB& b, bool add) {
            const int pt = a_sf & 7;
            const std::uint64_t atk = (pt == 1) ? pawn_attacks(a_sf < 8 ? 0 : 1, a_sq) : knight_attacks(a_sq);
            if (!(atk & (1ull << tgt))) return;
            const int attacked = b.piece_on[tgt];
            if (attacked < 0) return;
            emit_threat(a_sf, a_sq, tgt, attacked, add);
        };
        auto pp_index = [&](int orient, int persp, int cA, int fsq, int tsq, int cBc) -> std::uint32_t {
            const int fo = fsq ^ orient, to_o = tsq ^ orient;
            const int ca = cA ^ persp, cb = cBc ^ persp;
            const int id_a = 48 * ca + (fo - 8), id_b = 48 * cb + (to_o - 8);
            const int hi = std::max(id_a, id_b), lo = std::min(id_a, id_b);
            return (std::uint32_t)(hi * (hi - 1) / 2 + lo + PP_INDEX_BASE);
        };
        auto emit_pp = [&](const BB& b, int sq, int color, bool add, int exclude_sq) {
            const std::uint64_t wp = b.by[SF_W_PAWN], bp = b.by[SF_B_PAWN];
            std::uint64_t band = g_tab.pawn_pair[sq] & (wp | bp);
            while (band) {
                const int t = __builtin_ctzll(band); band &= band - 1;
                if (t == exclude_sq) continue;
                const int cB = (bp >> t) & 1;
                push(0, pp_index(wo, 0, color, sq, t, cB), add);
                push(1, pp_index(bo, 1, color, sq, t, cB), add);
            }
        };
        // 1) moved piece: old threats (bb0@from) subtracted, new (bb1@to) added
        emit_full(bb0, from, sf_moved, false);
        emit_full(bb1, to,   sf_moved, true);
        // 2) captured piece: its threats vanish
        if (cap) emit_full(bb0, to, ((cpi / 6) << 3) | ((cpi % 6) + 1), false);
        // 3) attackers of from/to. Non-sliders (knights/pawns) are occupancy-
        //    independent -> computed ONCE (identical in bb0/bb1, off the moved
        //    squares); kings never attack -> skipped. Sliders are relevant if they
        //    see from/to in EITHER board (occupancy changed only there), via the
        //    super-piece rays from from/to under each occupancy.
        const std::uint64_t Nb = bb0.by[2] | bb0.by[10];
        const std::uint64_t Wp = bb0.by[SF_W_PAWN], Bp = bb0.by[SF_B_PAWN];
        std::uint64_t nsatt = (knight_attacks(from) | knight_attacks(to)) & Nb;
        nsatt |= Wp & (pawn_attacks(1, from) | pawn_attacks(1, to));  // white pawns hitting from/to
        nsatt |= Bp & (pawn_attacks(0, from) | pawn_attacks(0, to));  // black pawns hitting from/to
        nsatt &= ~excl;
        const std::uint64_t BQ = bb0.by[3] | bb0.by[11] | bb0.by[5] | bb0.by[13];
        const std::uint64_t RQ = bb0.by[4] | bb0.by[12] | bb0.by[5] | bb0.by[13];
        std::uint64_t sliders =
            (((attacks_bb(3, from, bb0.occ) | attacks_bb(3, from, bb1.occ)
             | attacks_bb(3, to, bb0.occ)   | attacks_bb(3, to, bb1.occ)) & BQ)
           | ((attacks_bb(4, from, bb0.occ) | attacks_bb(4, from, bb1.occ)
             | attacks_bb(4, to, bb0.occ)   | attacks_bb(4, to, bb1.occ)) & RQ)) & ~excl;
        while (sliders) {
            const int s = __builtin_ctzll(sliders); sliders &= sliders - 1;
            const int sf = bb0.piece_on[s];      // slider exists in both (s not in excl)
            const int pt = sf & 7;
            const std::uint64_t a0 = attacks_bb(pt, s, bb0.occ);
            const std::uint64_t a1 = attacks_bb(pt, s, bb1.occ);
            std::uint64_t rel = (a0 ^ a1) | ((a0 & a1) & excl);  // gained/lost squares + from/to identity flip
            while (rel) {
                const int q = __builtin_ctzll(rel); rel &= rel - 1;
                if ((a0 >> q) & 1) { const int p0 = bb0.piece_on[q]; if (p0 >= 0) emit_threat(sf, s, q, p0, false); }
                if ((a1 >> q) & 1) { const int p1 = bb1.piece_on[q]; if (p1 >= 0) emit_threat(sf, s, q, p1, true); }
            }
        }
        while (nsatt) {
            const int s = __builtin_ctzll(nsatt); nsatt &= nsatt - 1;
            const int sf = bb0.piece_on[s];      // non-slider: same square/type in both
            emit_ns(s, sf, from, bb0, false);    // it threatened the mover at `from`
            emit_ns(s, sf, to,   bb0, false);    // and whatever was on `to` (capture/empty)
            emit_ns(s, sf, to,   bb1, true);     // now threatens the mover at `to`
            // (`from` is empty in bb1 -> nothing to add there)
        }
        // 4) pawn pairs: moved pawn's pairs move; captured pawn's vanish (dedup the
        //    shared P x P pair by excluding the mover's `from` square once).
        if (mt0 == 0) {
            emit_pp(bb0, from, mcolor, false, -1);
            emit_pp(bb1, to,   mcolor, true,  -1);
        }
        if (cap && (cpi % 6) == 0)
            emit_pp(bb0, to, cpi / 6, false, mt0 == 0 ? from : -1);
}

// FXZ: a pending normal move's board and column lists, made at the first read of its ply.
void fx_enumerate(Context& cx, int k) {
    Context::FxPend& e = cx.fx_pend[k];
    const BB& bb0 = cx.bb[k - 1];
    BB& bb1 = cx.bb[k];
    bb_apply(bb0, bb1, e.from, e.to, e.mcolor, e.mt0, e.cap, e.cpi);
    e.na[0] = e.na[1] = e.ns[0] = e.ns[1] = 0;
    delta_lists<std::int32_t>(bb0, bb1, e.from, e.to, e.mcolor, e.mt0, e.cap, e.cpi, g_net.l0w_fx_p, g_net.hl, true,
                              fx_adds(cx, e), fx_subs(cx, e), e.na, e.ns);
}

// FXZ: at the first read of a ply whose king crossed a bucket (same mirror), bring the mover's own base
// block over in the same single pass: with enough men, the two buckets' K1 sums (up to date with only
// the changed pieces) join the column lists; with few, each man's old and new column (2 per man is
// already cheaper than two cache passes there). int32 sums: the same integers either way.
void fx_rebucket(Context& cx, Context::FxPend& e, const BB& bb1) {
    const std::int32_t* const l0 = g_net.l0w_fx_p;
    const int hl = g_net.hl;
    const int mcolor = e.rb_color, from = e.from, to = e.to;
    if (__builtin_popcountll(bb1.occ) >= SC_FX_K1_MIN) {
        const int flip = (file_of(mcolor == 0 ? to : (to ^ 56)) > 3) ? 1 : 0;
        const int kb_old = king_bucket(mcolor == 0 ? from : (from ^ 56));
        const int kb_new = king_bucket(mcolor == 0 ? to : (to ^ 56));
        auto bring_up = [&](Context::FinnyFx& c, int ksq) {
            const std::int32_t** ra = reinterpret_cast<const std::int32_t**>(cx.ra);
            const std::int32_t** rs = reinterpret_cast<const std::int32_t**>(cx.rs);
            int nra = 0, nrs = 0;
            if (!c.valid) {
                std::memset(c.acc, 0, sizeof(c.acc));
                std::memset(c.by, 0, sizeof(c.by));
                c.valid = true;
            }
            for (int sf = 1; sf < 15; ++sf) {
                if ((sf & 7) == 0 || (sf & 7) == 7) continue;
                std::uint64_t gone = c.by[sf] & ~bb1.by[sf], added = bb1.by[sf] & ~c.by[sf];
                const int col = sf >> 3, t0 = (sf & 7) - 1;
                while (gone)  { const int sq = __builtin_ctzll(gone);  gone &= gone - 1;
                    rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, ksq) * hl; }
                while (added) { const int sq = __builtin_ctzll(added); added &= added - 1;
                    ra[nra++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, ksq) * hl; }
                c.by[sf] = bb1.by[sf];
                if (nra > 32 || nrs > 32) {   // flush (the scratch holds 40)
                    fused_apply_fx(c.acc, ra, nra, rs, nrs, hl);
                    nra = nrs = 0;
                }
            }
            if (nra + nrs) fused_apply_fx(c.acc, ra, nra, rs, nrs, hl);
        };
        Context::FinnyFx* const tab = cx.finny_fx.data() + (mcolor * 2 + flip) * kKingBuckets;
        bring_up(tab[kb_old], from);
        bring_up(tab[kb_new], to);
        fx_adds(cx, e)[mcolor][e.na[mcolor]++] = tab[kb_new].acc;
        fx_subs(cx, e)[mcolor][e.ns[mcolor]++] = tab[kb_old].acc;
    } else {
        std::uint64_t occ = bb1.occ;
        while (occ) {
            const int sq = __builtin_ctzll(occ); occ &= occ - 1;
            const int sf = bb1.piece_on[sq];
            const int col = sf >> 3, t0 = (sf & 7) - 1;
            fx_subs(cx, e)[mcolor][e.ns[mcolor]++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, from) * hl;  // old bucket
            fx_adds(cx, e)[mcolor][e.na[mcolor]++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, to)   * hl;  // new bucket
        }
    }
}

#if SC_Q_LAZY
// A pending int8 ply at its first read: the board, the lists and one src->dst pass per perspective; a king across a
// bucket (same mirror) then brings the mover's own base block over, as the eager path below does.
void q_apply(Context& cx, int k) {
    const Context::QPend& e = cx.q_pend[k];
    const int hl = g_net.hl;
    Acc& cur = cx.stack[k - 1];
    Acc& nxt = cx.stack[k];
    if (e.kind == 1) {
        std::memcpy(nxt.v[0], cur.v[0], hl * sizeof(std::int16_t));
        std::memcpy(nxt.v[1], cur.v[1], hl * sizeof(std::int16_t));
        cx.bb[k] = cx.bb[k - 1];
        return;
    }
    using WT = std::int8_t;
    const WT* const l0 = g_net.l0w_i8.data();
    const int from = e.from, to = e.to, mcolor = e.mcolor;
    const BB& bb0 = cx.bb[k - 1];
    BB& bb1 = cx.bb[k];
    bb_apply(bb0, bb1, from, to, mcolor, e.mt0, e.cap, e.cpi);
    const WT* (&addp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.addp);
    const WT* (&subp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.subp);
    int na[2] = {0, 0}, ns[2] = {0, 0};
    delta_lists<WT>(bb0, bb1, from, to, mcolor, e.mt0, e.cap, e.cpi, l0, hl, true, addp, subp, na, ns);
    fused_apply_src(nxt.v[0], cur.v[0], addp[0], na[0], subp[0], ns[0], hl);
    fused_apply_src(nxt.v[1], cur.v[1], addp[1], na[1], subp[1], ns[1], hl);
    if (!e.rb) return;
    if (__builtin_popcountll(bb1.occ) >= SC_NPS_K1_MIN) {
        const int flip = (file_of(mcolor == 0 ? to : (to ^ 56)) > 3) ? 1 : 0;
        const int kb_old = king_bucket(mcolor == 0 ? from : (from ^ 56));
        const int kb_new = king_bucket(mcolor == 0 ? to : (to ^ 56));
        auto refresh_finny = [&](Context::Finny& c, int ksq) {
            const WT** ra = reinterpret_cast<const WT**>(cx.ra);
            const WT** rs = reinterpret_cast<const WT**>(cx.rs);
            int nra = 0, nrs = 0;
            if (!c.valid) {
                std::memset(c.acc, 0, sizeof(c.acc));
                std::memset(c.by, 0, sizeof(c.by));
                c.valid = true;
            }
            for (int sf = 1; sf < 15; ++sf) {
                if ((sf & 7) == 0 || (sf & 7) == 7) continue;
                std::uint64_t gone = c.by[sf] & ~bb1.by[sf], added = bb1.by[sf] & ~c.by[sf];
                const int col = sf >> 3, t0 = (sf & 7) - 1;
                while (gone)  { const int sq = __builtin_ctzll(gone);  gone &= gone - 1;
                    rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, ksq) * hl; }
                while (added) { const int sq = __builtin_ctzll(added); added &= added - 1;
                    ra[nra++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, ksq) * hl; }
                c.by[sf] = bb1.by[sf];
                if (nra > 32 || nrs > 32) {   // flush (the scratch holds 40)
                    fused_apply(c.acc, ra, nra, rs, nrs, hl);
                    nra = nrs = 0;
                }
            }
            if (nra + nrs) fused_apply(c.acc, ra, nra, rs, nrs, hl);
        };
        Context::Finny& eo = cx.finny[mcolor][flip][kb_old];
        Context::Finny& en = cx.finny[mcolor][flip][kb_new];
        refresh_finny(eo, from);
        refresh_finny(en, to);
        std::int16_t* a = nxt.v[mcolor];
        for (int j = 0; j < hl; ++j)
            a[j] = static_cast<std::int16_t>(a[j] - eo.acc[j] + en.acc[j]);
    } else {
        const WT** ra = reinterpret_cast<const WT**>(cx.ra);
        const WT** rs = reinterpret_cast<const WT**>(cx.rs);
        int nra = 0, nrs = 0;
        std::uint64_t occ = bb1.occ;
        while (occ) {
            const int sq = __builtin_ctzll(occ); occ &= occ - 1;
            const int sf = bb1.piece_on[sq];
            const int col = sf >> 3, t0 = (sf & 7) - 1;
            rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, from) * hl;  // old bucket (king@from)
            ra[nra++] = l0 + (std::size_t)base_feat(mcolor, col, t0, sq, to)   * hl;  // new bucket (king@to)
        }
        fused_apply(nxt.v[mcolor], ra, nra, rs, nrs, hl);
    }
}
#endif

// Incremental delta, templated on accumulator/weight type so the intricate delta
// logic lives once. Quant instantiates <Acc,int8> (byte-identical to before, guarded
// by verify_acc); the float labeler instantiates <AccFx,int32> (fixed-point).
template <class AccT, class WT>
void acc_make_t(Context& cx, const Board& before, Move m, std::vector<AccT>& g_stk,
                const WT* l0, void (*refresh_fn)(AccT&, const Board&)) {
    constexpr bool FX = (sizeof(WT) != 1);  // int32 fixed-point float vs int8 quant
    constexpr bool LZ = FX && SC_FX_LAZY;   // FXZ: the float path defers its board, lists and column passes
    if (cx.ply < 0 || cx.ply + 1 >= ACC_STACK) { cx.ply = -1; return; }
    const int hl = g_net.hl;
    AccT& cur = g_stk[cx.ply];
    AccT& nxt = g_stk[cx.ply + 1];
    // fused-apply dispatch: int8/int16 (quant) vs int32/int32 (fixed-point float).
    auto fused = [&](auto* acc, const WT* const* add, int na, const WT* const* sub, int ns) {
        if constexpr (FX) fused_apply_fx(acc, add, na, sub, ns, hl);
        else              fused_apply(acc, add, na, sub, ns, hl);
    };
    // G4 single pass: dst = src +/- columns (quant path; the float labeler joins with FX2).
    [[maybe_unused]] auto fused_src = [&](auto* dst, const auto* src, const WT* const* add, int na,
                                          const WT* const* sub, int ns) {
        if constexpr (FX) fused_apply_src_fx(dst, src, add, na, sub, ns, hl);
        else              fused_apply_src(dst, src, add, na, sub, ns, hl);
    };
    const int from = m.from().index(), to = m.to().index();
    const int moved_pi = static_cast<int>(before.at(m.from()).internal());
    const int mcolor = moved_pi / 6, mt0 = moved_pi % 6;
    // Refresh is only needed when the moving side's king changes BUCKET or MIRROR
    // (e-file) side: base features are king-bucketed+mirrored, threats/pp are only
    // mirrored, and the king is never a threat attacker/target -- so a king move
    // that stays in the same bucket and mirror side is a plain incremental move
    // (the normal delta path handles the king-as-piece + slider blocking exactly).
    // Castle/promo/ep always refresh (multi-piece / index-space changes).
    bool special;
    bool king_rebucket = false;  // king crossed BUCKET but not mirror -> incremental + base fix
    if (m.typeOf() != Move::NORMAL) {
        special = true;
    } else if (mt0 == 5) {  // king normal move
        const int kp_f = (mcolor == 0) ? from : (from ^ 56);
        const int kp_t = (mcolor == 0) ? to : (to ^ 56);
        // Only a MIRROR (e-file) cross needs a full refresh: it flips the threat/pp
        // orientation for the mover's perspective. A pure bucket cross keeps the same
        // orientation, so it stays on the incremental path and only the mover's-own
        // base block is rebucketed afterwards (see king_rebucket below).
        const bool mirror_change = (file_of(kp_f) > 3) != (file_of(kp_t) > 3);
        special = g_no_finny || mirror_change;
        king_rebucket = !special && (king_bucket(kp_f) != king_bucket(kp_t));
    } else {
        special = false;
    }
    if (special) {
        // G8: make the move on the caller's own board and unmake it after the refresh,
        // instead of copying the Board (which deep-copies its undo stack) per refresh.
        // Every caller passes its mutable search board; the refresh reads exactly the
        // state a copy would have had, and unmakeMove restores the board in full.
        Board& b = const_cast<Board&>(before);
        if constexpr (FX) {
            // FXD: a castling, promotion, en-passant or mirror-crossing king move changes few of the
            // ~100 active columns of a perspective, except the mirror-crossing one's own. Diff the two
            // feature multisets and apply only the difference (int32 sums: the same integers as a
            // refresh); a perspective that changes more than it keeps is summed from the bias.
            if constexpr (LZ) fx_ensure(cx, cx.ply);
            const std::size_t bd = g_net.base_dims;
            int nf[2][2] = {{0, 0}, {0, 0}};
            auto collect = [&](const Board& bd_board, int w) {
                const int s = (bd_board.sideToMove() == Color::WHITE) ? 0 : 1;
                gather(bd_board, bd,
                       [&](std::size_t f) { cx.fxd[w][s][nf[w][s]++] = static_cast<std::uint32_t>(f); },
                       [&](std::size_t f) { cx.fxd[w][1 - s][nf[w][1 - s]++] = static_cast<std::uint32_t>(f); });
            };
            collect(before, 0);
            b.makeMove(m);
            collect(b, 1);
            cx.bb[cx.ply + 1] = BB(b);  // keep the incremental BB chain in sync
            b.unmakeMove(m);
            const WT* (&addp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.addp);
            const WT* (&subp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.subp);
            for (int p = 0; p < 2; ++p) {
                const std::uint32_t* const f0 = cx.fxd[0][p];
                const std::uint32_t* const f1 = cx.fxd[1][p];
                std::uint8_t* const cnt = cx.fxd_count.data();   // zero between uses
                int na = 0, ns = 0;
                for (int k = 0; k < nf[0][p]; ++k) ++cnt[f0[k]];
                for (int k = 0; k < nf[1][p]; ++k) {
                    if (cnt[f1[k]]) --cnt[f1[k]];
                    else addp[p][na++] = l0 + static_cast<std::size_t>(f1[k]) * hl;
                }
                for (int k = 0; k < nf[0][p]; ++k)
                    if (cnt[f0[k]]) { --cnt[f0[k]]; subp[p][ns++] = l0 + static_cast<std::size_t>(f0[k]) * hl; }
                if (na + ns < nf[1][p]) {
                    fused_src(nxt.v[p], cur.v[p], addp[p], na, subp[p], ns);
                } else {
                    for (int k = 0; k < nf[1][p]; ++k) addp[p][k] = l0 + static_cast<std::size_t>(f1[k]) * hl;
                    fused_src(nxt.v[p], g_net.l0b_fx.data(), addp[p], nf[1][p], subp[p], 0);
                }
            }
        } else {
            b.makeMove(m);
            refresh_fn(nxt, b);
            cx.bb[cx.ply + 1] = BB(b);  // keep the incremental BB chain in sync
            b.unmakeMove(m);
#if SC_Q_LAZY
            cx.q_ok[cx.ply + 1] = 1;   // a refresh does not read the parent, which can stay pending
#endif
        }
        if constexpr (LZ) cx.fx_ok[cx.ply + 1] = 1;
    } else {
        // Single pass (x86 campaign G4, +3.7% all phases): the base columns are deferred into
        // the threat add/sub lists and ONE src->dst fused pass per perspective reads cur and
        // writes nxt -- no memcpy, no separate base pass. Integer sums are order-independent
        // -> bit-identical. The probes (NOUPDATE/NOCOLS) keep the two-pass form.
        // The float labeler takes the same single pass (int32 sums -> bit-identical; self-play
        // campaign FX2).
        const bool single_pass = !g_no_update && !g_no_cols;
#if SC_Q_LAZY
        if constexpr (!FX) {
            if (single_pass) {   // record the move and stop; see q_apply
                Context::QPend& e = cx.q_pend[cx.ply + 1];
                const chess::Piece cp = before.at(m.to());
                const bool c = (cp != chess::Piece::NONE);
                e.kind = 0;
                e.from = static_cast<std::int8_t>(from);
                e.to = static_cast<std::int8_t>(to);
                e.mcolor = static_cast<std::int8_t>(mcolor);
                e.mt0 = static_cast<std::int8_t>(mt0);
                e.cap = c;
                e.cpi = c ? static_cast<std::int8_t>(cp.internal()) : std::int8_t(0);
                e.rb = king_rebucket;
                cx.q_ok[cx.ply + 1] = 0;
                cx.ply++;
                return;
            }
            cx.q_ok[cx.ply + 1] = 1;
            q_ensure(cx, cx.ply);   // the probes read cur up front
        }
#endif
        if constexpr (LZ) {
            if (single_pass) {
                // FXZ: record the move and stop. Its board, its column lists and the passes all wait for
                // the first read of this ply; a ply popped unread costs nothing more.
                Context::FxPend& e = cx.fx_pend[cx.ply + 1];
                const chess::Piece cp = before.at(m.to());
                e.kind = 0;
                e.from = from;
                e.to = to;
                e.mcolor = mcolor;
                e.mt0 = mt0;
                e.cap = (cp != chess::Piece::NONE);
                e.cpi = e.cap ? static_cast<int>(cp.internal()) : 0;
                e.rb_color = king_rebucket ? mcolor : -1;
                cx.fx_ok[cx.ply + 1] = 0;
                cx.ply++;
                return;
            }
            cx.fx_ok[cx.ply + 1] = 1;
            fx_ensure(cx, cx.ply);   // the probes read cur up front
        }
        if (!single_pass) {
            std::memcpy(nxt.v[0], cur.v[0], hl * sizeof(nxt.v[0][0]));
            std::memcpy(nxt.v[1], cur.v[1], hl * sizeof(nxt.v[0][0]));
        }
        const int kabs[2] = {before.kingSq(Color::WHITE).index(), before.kingSq(Color::BLACK).index()};
        const chess::Piece capp = before.at(m.to());
        const bool cap = (capp != chess::Piece::NONE);
        const int cpi = cap ? static_cast<int>(capp.internal()) : 0;
        // ---- base delta (piece-square): one fused pass per persp, or deferred into the
        // threat lists when single_pass (G4) ----
        const WT* base_add[2];
        const WT* base_sub[2][2];
        const int nbase_sub = cap ? 2 : 1;
        for (int p = 0; p < 2; ++p) {
            base_add[p]    = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, to, kabs[p])) * hl;
            base_sub[p][0] = l0 + static_cast<std::size_t>(base_feat(p, mcolor, mt0, from, kabs[p])) * hl;
            if (cap) base_sub[p][1] = l0 + static_cast<std::size_t>(base_feat(p, cpi / 6, cpi % 6, to, kabs[p])) * hl;
            if (!single_pass) fused(nxt.v[p], &base_add[p], 1, base_sub[p], nbase_sub);
        }
        // ---- threat + pp delta (exact: collect before/after feature sets over
        // the affected pieces, then apply only the multiset difference) ----
        // Incremental board bitboards: the before-state is already on the stack
        // (cx.bb[cx.ply]), so bb0 is a reference -- no BB rebuild from board.at().
        // Form the after-move child directly in the next slot.
        const BB& bb0 = cx.bb[cx.ply];
        BB& bb1 = cx.bb[cx.ply + 1];
        if (g_verify) {  // the stack BB must equal a fresh recompute of `before`
            BB chk(before);
            if (bb0.occ != chk.occ || bb0.wk != chk.wk || bb0.bk != chk.bk
                || std::memcmp(bb0.by, chk.by, sizeof(chk.by)) != 0
                || std::memcmp(bb0.piece_on, chk.piece_on, sizeof(chk.piece_on)) != 0)
                std::fprintf(stderr, "[nnue] BB MISMATCH ply=%d\n", cx.ply);
        }
        bb_apply(bb0, bb1, from, to, mcolor, mt0, cap, cpi);
        if (g_no_update) { cx.ply++; return; }  // cost probe: skip the delta (BB kept)
      {   // direct delta
        const WT* (&addp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.addp);
        const WT* (&subp)[2][1024] = *reinterpret_cast<const WT* (*)[2][1024]>(&cx.subp);
        int na[2] = {0, 0}, ns[2] = {0, 0};
        delta_lists<WT>(bb0, bb1, from, to, mcolor, mt0, cap, cpi, l0, hl, single_pass, addp, subp, na, ns);
        if (g_prefetch && !SC_NPS_P1)
            for (int p = 0; p < 2; ++p) {
                for (int k = 0; k < na[p]; ++k)
                    for (int t = 0; t < SC_NPS_PF_N; ++t)
                        __builtin_prefetch(reinterpret_cast<const char*>(addp[p][k]) + t * kPfStride, 0, SC_NPS_PF_LOC);
                for (int k = 0; k < ns[p]; ++k)
                    for (int t = 0; t < SC_NPS_PF_N; ++t)
                        __builtin_prefetch(reinterpret_cast<const char*>(subp[p][k]) + t * kPfStride, 0, SC_NPS_PF_LOC);
            }
        if (!g_no_cols) {  // NOCOLS probe: keep enumeration, skip the column adds
            if (single_pass) {   // G4: cur -> nxt in one pass (base + threat + pp columns)
                fused_src(nxt.v[0], cur.v[0], addp[0], na[0], subp[0], ns[0]);
                fused_src(nxt.v[1], cur.v[1], addp[1], na[1], subp[1], ns[1]);
            } else {
                fused(nxt.v[0], addp[0], na[0], subp[0], ns[0]);
                fused(nxt.v[1], addp[1], na[1], subp[1], ns[1]);
            }
        }
      }
        // King crossed a bucket (same mirror): the mover's-own perspective still has
        // every piece indexed under the OLD king bucket. Rebucket v[mcolor] in place
        // -- sub each piece's old-bucket base column, add its new-bucket column. The
        // opponent perspective and all threats/pp were already correct on the
        // incremental path (same mirror -> same orientation; king isn't a threat).
        if (king_rebucket) {
          bool rebucketed = false;
          // With few men the direct form (2 columns per man) is already cheaper than two cache
          // passes; the caches pay where the board is full.
          if constexpr (!FX) if (__builtin_popcountll(bb1.occ) >= SC_NPS_K1_MIN) {
            rebucketed = true;
            // Bring the old and the new bucket's cache to the current pieces, then swap them in.
            const int flip = (file_of(mcolor == 0 ? to : (to ^ 56)) > 3) ? 1 : 0;
            const int kb_old = king_bucket(mcolor == 0 ? from : (from ^ 56));
            const int kb_new = king_bucket(mcolor == 0 ? to : (to ^ 56));
            auto refresh_finny = [&](Context::Finny& e, int ksq) {
                const WT** ra = reinterpret_cast<const WT**>(cx.ra);
                const WT** rs = reinterpret_cast<const WT**>(cx.rs);
                int nra = 0, nrs = 0;
                if (!e.valid) {
                    std::memset(e.acc, 0, sizeof(e.acc));
                    std::memset(e.by, 0, sizeof(e.by));
                    e.valid = true;
                }
                for (int sf = 1; sf < 15; ++sf) {
                    if ((sf & 7) == 0 || (sf & 7) == 7) continue;
                    std::uint64_t gone = e.by[sf] & ~bb1.by[sf], added = bb1.by[sf] & ~e.by[sf];
                    const int c = sf >> 3, t0 = (sf & 7) - 1;
                    while (gone)  { const int sq = __builtin_ctzll(gone);  gone &= gone - 1;
                        rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, ksq) * hl; }
                    while (added) { const int sq = __builtin_ctzll(added); added &= added - 1;
                        ra[nra++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, ksq) * hl; }
                    e.by[sf] = bb1.by[sf];
                    if (nra > 32 || nrs > 32) {   // flush (the scratch holds 40)
                        fused(e.acc, ra, nra, rs, nrs);
                        nra = nrs = 0;
                    }
                }
                if (nra + nrs) fused(e.acc, ra, nra, rs, nrs);
            };
            Context::Finny& eo = cx.finny[mcolor][flip][kb_old];
            Context::Finny& en = cx.finny[mcolor][flip][kb_new];
            refresh_finny(eo, from);
            refresh_finny(en, to);
            std::int16_t* a = nxt.v[mcolor];
            for (int j = 0; j < hl; ++j)
                a[j] = static_cast<std::int16_t>(a[j] - eo.acc[j] + en.acc[j]);
          }
          if (!rebucketed) {
            const WT** ra = reinterpret_cast<const WT**>(cx.ra);
            const WT** rs = reinterpret_cast<const WT**>(cx.rs);
            int nra = 0, nrs = 0;
            std::uint64_t occ = bb1.occ;
            while (occ) {
                const int sq = __builtin_ctzll(occ); occ &= occ - 1;
                const int sf = bb1.piece_on[sq];
                const int c = sf >> 3, t0 = (sf & 7) - 1;
                rs[nrs++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, from) * hl;  // old bucket (king@from)
                ra[nra++] = l0 + (std::size_t)base_feat(mcolor, c, t0, sq, to)   * hl;  // new bucket (king@to)
            }
            fused(nxt.v[mcolor], ra, nra, rs, nrs);
          }
        }
    }
    cx.ply++;
}

// Dispatch acc_make to the active accumulator (quant int16 / float fixed-point int32).
void acc_make(Context& cx, const Board& before, Move m) {
    if (g_net.quant)
        acc_make_t<Acc, std::int8_t>(cx, before, m, cx.stack, g_net.l0w_i8.data(), refresh_into);
    else
        acc_make_t<AccFx, std::int32_t>(cx, before, m, cx.stack_fx, g_net.l0w_fx_p, refresh_into_fx);
}

void acc_unmake(Context& cx) {
    if (cx.ply > 0) cx.ply--;
}

void acc_clear(Context& cx) { cx.ply = -1; }


void acc_make_null(Context& cx) {
    if (cx.ply < 0 || cx.ply + 1 >= ACC_STACK) { cx.ply = -1; return; }
    const int hl = g_net.hl;
    if (g_net.quant) {
#if SC_Q_LAZY
        cx.q_pend[cx.ply + 1].kind = 1;   // the copy and the board wait for the first read
        cx.q_ok[cx.ply + 1] = 0;
        cx.ply++;
        return;
#endif
        std::memcpy(cx.stack[cx.ply + 1].v[0], cx.stack[cx.ply].v[0], hl * sizeof(std::int16_t));
        std::memcpy(cx.stack[cx.ply + 1].v[1], cx.stack[cx.ply].v[1], hl * sizeof(std::int16_t));
    } else if constexpr (SC_FX_LAZY) {   // FXZ: an empty column list = the copy, made on first read
        Context::FxPend& e = cx.fx_pend[cx.ply + 1];
        e.kind = 1;
        e.na[0] = e.na[1] = e.ns[0] = e.ns[1] = 0;
        e.rb_color = -1;
        cx.fx_ok[cx.ply + 1] = 0;
        cx.ply++;
        return;
    } else {
        std::memcpy(cx.stack_fx[cx.ply + 1].v[0], cx.stack_fx[cx.ply].v[0], hl * sizeof(std::int32_t));
        std::memcpy(cx.stack_fx[cx.ply + 1].v[1], cx.stack_fx[cx.ply].v[1], hl * sizeof(std::int32_t));
    }
    cx.bb[cx.ply + 1] = cx.bb[cx.ply];  // piece placement unchanged by a null move
    cx.ply++;
}
void acc_unmake_null(Context& cx) {
    if (cx.ply > 0) cx.ply--;
}

// Context-free forms: the calling thread's own context (UCI debug commands, tools).
Value evaluate(const Board& board) { return evaluate(tl_context(), board); }
Value evaluate_pass(const Board& board) { return evaluate_pass(tl_context(), board); }
void acc_reset(const Board& board) { acc_reset(tl_context(), board); }
void acc_make(const Board& before, Move m) { acc_make(tl_context(), before, m); }
void acc_unmake() { acc_unmake(tl_context()); }
void acc_clear() { acc_clear(tl_context()); }
void acc_make_null() { acc_make_null(tl_context()); }
void acc_unmake_null() { acc_unmake_null(tl_context()); }

}  // namespace engine::nnue
