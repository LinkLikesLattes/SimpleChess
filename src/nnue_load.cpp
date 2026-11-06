// -----------------------------------------------------------------------------
// nnue_load.cpp -- reading an SCN4 (float) / SCN5 (int8) net file into g_net and laying its weights out
// for inference: the threat-index tables, the transposed / bucket-major / permuted weight copies, the
// fixed-point FT of a float net (shared across processes, FXS), the Windows large-page allocator.
// Runs once per EvalFile; nothing here is on the search's hot path.
// -----------------------------------------------------------------------------
#include "nnue_internal.hpp"

#include <fstream>
#include <istream>
#include <streambuf>
#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#if defined(_WIN32)
// D2: back the ~35 MB feature-transformer weights (l0w_i8) with 2 MB large pages to cut
// TLB misses on the scattered 512 B feature-row gathers. Only allocations >= 2 MB take
// this path, so the small L1/L2/L3 weight vectors keep operator new. A tiny registry
// records which pointers are large-page-backed so deallocate() frees them correctly.
// The stored weights are byte-identical either way -> bit-identical.
#include <mutex>
#include <unordered_set>
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace engine::nnue {
namespace {
std::mutex             g_lp_mu;
std::unordered_set<void*>& g_lp_set() { static std::unordered_set<void*> s; return s; }
bool sc_enable_lock_pages_nnue() {
    HANDLE tok = nullptr;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok))
        return false;
    TOKEN_PRIVILEGES tp{};
    tp.PrivilegeCount           = 1;
    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValue(nullptr, SE_LOCK_MEMORY_NAME, &tp.Privileges[0].Luid) &&
              AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) &&
              GetLastError() == ERROR_SUCCESS;
    CloseHandle(tok);
    return ok;
}
}  // namespace
void* sc_nnue_large_alloc(std::size_t bytes) {
    static const bool priv = sc_enable_lock_pages_nnue();
    if (!priv) return nullptr;
    const std::size_t gran = GetLargePageMinimum();
    if (gran == 0) return nullptr;
    const std::size_t rounded = ((bytes + gran - 1) / gran) * gran;
    void* p = VirtualAlloc(nullptr, rounded, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                           PAGE_READWRITE);
    if (p) { std::lock_guard<std::mutex> lk(g_lp_mu); g_lp_set().insert(p); }
    return p;
}
bool sc_nnue_large_free(void* p) {
    std::lock_guard<std::mutex> lk(g_lp_mu);
    auto& s = g_lp_set();
    auto it = s.find(p);
    if (it == s.end()) return false;
    s.erase(it);
    VirtualFree(p, 0, MEM_RELEASE);
    return true;
}
}  // namespace engine::nnue
#endif

namespace engine::nnue {

Tables   g_tab;
Net      g_net;
unsigned g_net_gen = 0;
#if SC_NNUE_PROBES
#define SC_PROBE_DEF(type, name, init) type name = init;
SC_NNUE_PROBE_LIST(SC_PROBE_DEF)
#undef SC_PROBE_DEF
#endif

void build_tables() {
    if (g_tab.built) return;
    Tables& t = g_tab;
    const int all_pieces[12] = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14};

    std::array<std::uint32_t, 16> piece_off{};
    std::array<std::uint32_t, 16> cum_off{};
    std::uint32_t cumulative_offset = 0;
    for (int piece : all_pieces) {
        const int pt = piece & 7;
        std::uint32_t cpo = 0;
        for (int from = 0; from < 64; ++from) {
            t.offsets[piece][from] = cpo;
            if (pt != SF_W_PAWN)
                cpo += popcount64(pseudo_attacks_type(pt, from));
            else if (from >= 8 && from <= 55)
                cpo += popcount64(pseudo_for_index(piece, from));
        }
        piece_off[piece] = cpo;
        cum_off[piece] = cumulative_offset;
        cumulative_offset += static_cast<std::uint32_t>(NUM_VALID_TARGETS[piece]) * cpo;
    }

    for (int attacker : all_pieces)
        for (int attacked : all_pieces) {
            const bool enemy = (attacker ^ attacked) == 8;
            const int at = attacker & 7, dt = attacked & 7;
            const int map_val = MAP[at - 1][dt - 1];
            const bool semi_excluded = at == dt && (enemy || at != SF_W_PAWN);
            const int color_attacked = (attacked >> 3) & 1;
            const long feature = static_cast<long>(cum_off[attacker])
                + (static_cast<long>(color_attacked) * (NUM_VALID_TARGETS[attacker] / 2) + map_val)
                    * static_cast<long>(piece_off[attacker]);
            const bool excluded = map_val < 0;
            t.lut1[attacker][attacked][0] = excluded ? THREAT_DIMS : static_cast<std::uint32_t>(feature);
            t.lut1[attacker][attacked][1] =
                (excluded || semi_excluded) ? THREAT_DIMS : static_cast<std::uint32_t>(feature);
        }

    for (int piece : all_pieces)
        for (int from = 0; from < 64; ++from) {
            const std::uint64_t attacks = pseudo_for_index(piece, from);
            t.ao[piece][from] = {attacks, t.offsets[piece][from], 0};
            for (int to = 0; to < 64; ++to) {
                const std::uint64_t mask = (to == 0) ? 0 : ((1ull << to) - 1);
                t.lut2[piece][from][to] = static_cast<std::uint8_t>(popcount64(mask & attacks));
            }
        }

    for (int s = 0; s < 64; ++s) {
        const int f = file_of(s);
        std::uint64_t files = 0;
        for (int ff = f - 1; ff <= f + 1; ++ff)
            if (ff >= 0 && ff < 8)
                for (int r = 0; r < 8; ++r) files |= 1ull << (r * 8 + ff);
        std::uint64_t rank27 = 0;
        for (int r = 1; r < 7; ++r) rank27 |= 0xFFull << (r * 8);
        t.pawn_pair[s] = files & rank27 & ~(1ull << s);
    }

    t.built = true;
}

// FXS: the fixed-point FT columns of a float net, from (or into) a cache file shared by every engine
// process: $TMPDIR/simplechess-fx-<key>.i32, key = FNV-1a over the dimensions, the fixed-point shift and
// the float weights' bits. Under an exclusive flock the first process converts the weights exactly as
// the private path does, writes them to a temporary file and renames it into place; every process then
// maps the file read-only and shared, so the page cache holds one copy for all of them. Returns null on
// any failure (no TMPDIR, disk full, a short file): the caller builds its private copy instead.
//
// Windows: the first process publishes the columns in a named pagefile-backed section
// (SEC_LARGE_PAGES when the lock-pages privilege allows), named by a hash of the int32 columns
// themselves; the others open it under a named mutex and map it read-only only after a full memcmp
// against their own conversion (a section from another build or layout is never used). Nothing
// outlives the processes. SCNNUE_NOFXSHARE=1 disables it at run time on both (A/B on one binary);
// SCNNUE_FXSHARE_LOG=1 prints which path was taken to stderr.
std::shared_ptr<const std::int32_t> fx_share_map([[maybe_unused]] const std::vector<float>& l0w,
                                                 [[maybe_unused]] std::size_t ti, [[maybe_unused]] std::size_t hl) {
#if defined(_WIN32)
    const std::size_t nw = ti * hl, bytes = nw * sizeof(std::int32_t);
    if (l0w.size() != nw || nw == 0 || std::getenv("SCNNUE_NOFXSHARE")) return nullptr;
    std::vector<std::int32_t> fx(nw);
    for (std::size_t i = 0; i < nw; ++i)
        fx[i] = static_cast<std::int32_t>(std::lround(double(l0w[i]) * FX_S));
    std::uint64_t h = 1469598103934665603ull ^ bytes;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(fx.data());
    for (std::size_t i = 0; i + 8 <= bytes; i += 8) {
        std::uint64_t x;
        std::memcpy(&x, b + i, 8);
        h = (h ^ x) * 0x100000001b3ull;
        h ^= h >> 29;
    }
    char name[96], mname[104];
    std::snprintf(name, sizeof name, "Local\\SimpleChessFX_%016llx_%llu",
                  static_cast<unsigned long long>(h), static_cast<unsigned long long>(bytes));
    std::snprintf(mname, sizeof mname, "%s_m", name);
    HANDLE mtx = CreateMutexA(nullptr, FALSE, mname);
    if (!mtx) return nullptr;
    const DWORD wr = WaitForSingleObject(mtx, 120000);
    if (wr != WAIT_OBJECT_0 && wr != WAIT_ABANDONED) { CloseHandle(mtx); return nullptr; }
    void* view = nullptr;
    bool created = false, large = false;
    HANDLE sec = OpenFileMappingA(FILE_MAP_READ, FALSE, name);
    if (sec) {
        view = MapViewOfFile(sec, FILE_MAP_READ | FILE_MAP_LARGE_PAGES, 0, 0, 0);
        large = view != nullptr;
        if (!view) view = MapViewOfFile(sec, FILE_MAP_READ, 0, 0, 0);
        if (view && std::memcmp(view, fx.data(), bytes) != 0) { UnmapViewOfFile(view); view = nullptr; }
    } else {
        static const bool priv = sc_enable_lock_pages_nnue();
        const std::size_t gran = GetLargePageMinimum();
        if (priv && gran) {
            const std::size_t r = (bytes + gran - 1) / gran * gran;
            sec = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT | SEC_LARGE_PAGES,
                                     static_cast<DWORD>(r >> 32), static_cast<DWORD>(r), name);
            if (sec) {
                view = MapViewOfFile(sec, FILE_MAP_WRITE | FILE_MAP_LARGE_PAGES, 0, 0, 0);
                if (!view) view = MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, 0);
                if (!view) { CloseHandle(sec); sec = nullptr; }
                large = view != nullptr;
            }
        }
        if (!sec) {
            sec = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE | SEC_COMMIT,
                                     static_cast<DWORD>(bytes >> 32), static_cast<DWORD>(bytes), name);
            if (sec) view = MapViewOfFile(sec, FILE_MAP_WRITE, 0, 0, 0);
        }
        if (view) { std::memcpy(view, fx.data(), bytes); created = true; }
    }
    ReleaseMutex(mtx);
    CloseHandle(mtx);
    if (std::getenv("SCNNUE_FXSHARE_LOG"))
        std::fprintf(stderr, "FXS: %s %s pages\n", view ? (created ? "created" : "opened") : "private",
                     large ? "large" : "small");
    if (!view) { if (sec) CloseHandle(sec); return nullptr; }
    return std::shared_ptr<const std::int32_t>(static_cast<const std::int32_t*>(view),
                                               [sec](const std::int32_t* v) { UnmapViewOfFile(v); CloseHandle(sec); });
#else
    const std::size_t nw = ti * hl, bytes = nw * sizeof(std::int32_t);
    if (l0w.size() != nw || nw == 0 || std::getenv("SCNNUE_NOFXSHARE")) return nullptr;
    std::uint64_t key = 1469598103934665603ull;
    auto mix = [&key](std::uint64_t v) { key = (key ^ v) * 1099511628211ull; };
    mix(ti); mix(hl); mix(FX_SHIFT);
    for (float w : l0w) { std::uint32_t u; std::memcpy(&u, &w, 4); mix(u); }
    const char* dir = std::getenv("TMPDIR");
    std::string path = std::string(dir && *dir ? dir : "/tmp");
    if (path.back() != '/') path += '/';
    char name[48];
    std::snprintf(name, sizeof(name), "simplechess-fx-%016llx.i32", static_cast<unsigned long long>(key));
    path += name;

    const int lfd = ::open((path + ".lock").c_str(), O_CREAT | O_RDWR, 0644);
    if (lfd < 0) return nullptr;
    void* map = MAP_FAILED;
    if (::flock(lfd, LOCK_EX) == 0) {
        int fd = ::open(path.c_str(), O_RDONLY);
        struct stat st{};
        if (fd >= 0 && (::fstat(fd, &st) != 0 || static_cast<std::size_t>(st.st_size) != bytes)) {
            ::close(fd);
            fd = -1;
        }
        if (fd < 0) {   // first process for this net: convert and publish
            std::vector<std::int32_t> fx(nw);
            for (std::size_t i = 0; i < nw; ++i)
                fx[i] = static_cast<std::int32_t>(std::lround(double(l0w[i]) * FX_S));
            const std::string tmp = path + "." + std::to_string(::getpid()) + ".tmp";
            const int wfd = ::open(tmp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0644);
            bool ok = wfd >= 0;
            const char* src = reinterpret_cast<const char*>(fx.data());
            for (std::size_t done = 0; ok && done < bytes;) {
                const ssize_t w = ::write(wfd, src + done, bytes - done);
                if (w <= 0) ok = false;
                else done += static_cast<std::size_t>(w);
            }
            if (wfd >= 0 && ::close(wfd) != 0) ok = false;
            if (ok && ::rename(tmp.c_str(), path.c_str()) == 0) fd = ::open(path.c_str(), O_RDONLY);
            else ::unlink(tmp.c_str());
        }
        if (fd >= 0) {
            map = ::mmap(nullptr, bytes, PROT_READ, MAP_SHARED, fd, 0);
            ::close(fd);
        }
        ::flock(lfd, LOCK_UN);
    }
    ::close(lfd);
    if (map == MAP_FAILED) return nullptr;
    return std::shared_ptr<const std::int32_t>(static_cast<const std::int32_t*>(map),
                                               [bytes](const std::int32_t* p) { ::munmap(const_cast<std::int32_t*>(p), bytes); });
#endif
}

namespace {

template <class Vec>   // std::vector<float> or the 64-B-aligned WVec<float> (G3)
bool read_arr(std::istream& f, Vec& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()),
           static_cast<std::streamsize>(n) * static_cast<std::streamsize>(sizeof(typename Vec::value_type)));
    return static_cast<bool>(f);
}
bool read_i16(std::istream& f, std::vector<std::int16_t>& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n) * 2);
    return static_cast<bool>(f);
}
template <class Vec>   // std::vector<int8_t> or the 64-B-aligned WVec<int8_t> (G3)
bool read_i8(std::istream& f, Vec& dst, std::uint32_t expect) {
    std::uint32_t n = 0;
    f.read(reinterpret_cast<char*>(&n), 4);
    if (!f || n != expect) return false;
    dst.resize(n);
    f.read(reinterpret_cast<char*>(dst.data()), static_cast<std::streamsize>(n));
    return static_cast<bool>(f);
}

}  // namespace

static bool load_stream(std::istream& f) {

    char magic[4];
    f.read(magic, 4);
    if (!f) return false;
    const bool is_scn4 = std::memcmp(magic, "SCN4", 4) == 0;  // float
    const bool is_scn5 = std::memcmp(magic, "SCN5", 4) == 0;  // quantised (int8 FT)
    if (!is_scn4 && !is_scn5) return false;

    std::uint32_t hdr[5];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    if (!f) return false;
    if ((is_scn4 && hdr[0] != 4) || (is_scn5 && hdr[0] != 5)) return false;  // version

    Net n;
    n.quant = is_scn5;
    n.hl = int(hdr[1]);
    n.input_buckets = int(hdr[2]);
    n.l2 = int(hdr[3]);
    n.out_buckets = int(hdr[4]);
    // Only the shape the eval code is written for loads (a self-consistent file of another shape
    // would otherwise load "successfully" and then index out of range or divide by zero):
    //   input buckets == the king-bucket count of BUCKET_LAYOUT (king_bucket() returns 0..9),
    //   hl a multiple of 64 up to MAX_HL (the accumulator and pairwise SIMD loops; stack buffers),
    //   l2 a multiple of 16 up to MAX_L2 (the int8 dot loops step 16; stack buffers),
    //   1..32 output buckets (the output bucket divides the 32 men by the count).
    // The bounds also keep every array size below in 32 bits. Header fields are unsigned, so a
    // huge value turns negative here and fails the lower bounds.
    if (n.input_buckets != kKingBuckets || n.hl < 64 || n.hl > MAX_HL || n.hl % 64 != 0 ||
        n.l2 < 16 || n.l2 > MAX_L2 || n.l2 % 16 != 0 || n.out_buckets < 1 || n.out_buckets > 32)
        return false;
    const std::uint32_t HL = hdr[1], IB = hdr[2], L2 = hdr[3], OB = hdr[4];
    {   // output bucket of each piece count: (men - 2) / ceil(32 / OB), clamped to [0, OB)
        const int ob = n.out_buckets, divisor = (32 + ob - 1) / ob;
        for (int men = 0; men < 65; ++men)
            n.obucket[men] = static_cast<std::uint8_t>(std::clamp((men - 2) / divisor, 0, ob - 1));
    }
    n.base_dims = 768ull * IB;
    n.total_inputs = n.base_dims + THREAT_DIMS + PP_DIMS;
    const std::uint32_t TI = static_cast<std::uint32_t>(n.total_inputs);

    bool ok;
    if (is_scn4) {
        ok = read_arr(f, n.l0w, TI * HL)
          && read_arr(f, n.l0b, HL)
          && read_arr(f, n.l1w, OB * L2 * HL)
          && read_arr(f, n.l1b, OB * L2)
          && read_arr(f, n.l2w, OB * 32u * L2)
          && read_arr(f, n.l2b, OB * 32u)
          && read_arr(f, n.l3w, OB * 32u)
          && read_arr(f, n.l3b, OB);
    } else {
        ok = read_i8(f, n.l0w_i8, TI * HL)
          && read_i16(f, n.l0b_i, HL)
          && read_i8(f, n.l1w_i, OB * L2 * HL)
          && read_arr(f, n.l1b, OB * L2)
          && read_arr(f, n.l2w, OB * 32u * L2)
          && read_arr(f, n.l2b, OB * 32u)
          && read_arr(f, n.l3w, OB * 32u)
          && read_arr(f, n.l3b, OB);
    }
    if (!ok) return false;

    if (n.quant) {  // transpose L1 to [bucket*L2 + o][HL] (HL contiguous) for sdot
        const std::size_t OBL2 = static_cast<std::size_t>(OB) * L2;
        n.l1w_dot.resize(OBL2 * HL);
        for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
            for (std::uint32_t o = 0; o < L2; ++o)
                for (std::uint32_t i = 0; i < HL; ++i)
                    n.l1w_dot[(bkt * L2 + o) * HL + i] = n.l1w_i[static_cast<std::size_t>(i) * OBL2 + bkt * L2 + o];
    } else {  // float labeler: fixed-point FT weights/bias for the incremental accumulator
        const std::size_t NW = static_cast<std::size_t>(TI) * HL;
        if constexpr (SC_FX_SHARE) n.l0w_fx_shared = fx_share_map(n.l0w, TI, HL);
        if (!n.l0w_fx_shared) {
            n.l0w_fx.resize(NW);
            for (std::size_t i = 0; i < NW; ++i)
                n.l0w_fx[i] = static_cast<std::int32_t>(std::lround(double(n.l0w[i]) * FX_S));
        }
        n.l0w_fx_p = n.l0w_fx_shared ? n.l0w_fx_shared.get() : n.l0w_fx.data();
        n.l0b_fx.resize(HL);
        for (std::uint32_t j = 0; j < HL; ++j)
            n.l0b_fx[j] = static_cast<std::int32_t>(std::lround(double(n.l0b[j]) * FX_S));
        // eval_float now reads the fixed-point accumulator; the float FT weights/bias
        // are no longer needed -> free them (l0w_fx int32 replaces l0w float: net-neutral).
        std::vector<float>().swap(n.l0w);
        std::vector<float>().swap(n.l0b);
    }

    // L3 weights transposed to [bucket][32] so finish_body reads a contiguous row instead of a
    // stride-OB gather (both net formats; the sequential fma chain is unchanged). x86 campaign A6.
    n.l3w_t.resize(static_cast<std::size_t>(OB) * 32);
    for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
        for (std::uint32_t i = 0; i < 32; ++i)
            n.l3w_t[static_cast<std::size_t>(bkt) * 32 + i] = n.l3w[static_cast<std::size_t>(i) * OB + bkt];
    // L2 weights bucket-major [(b*L2 + i)*32 + o] so the AVX2 matvec walks 4 KB contiguous
    // per bucket instead of 32 rows at a 1 KB stride (same fma per lane, same i order). x86 campaign G2.
    n.l2w_bm.resize(static_cast<std::size_t>(OB) * L2 * 32);
    for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
        for (std::uint32_t i = 0; i < L2; ++i)
            for (std::uint32_t o = 0; o < 32; ++o)
                n.l2w_bm[(static_cast<std::size_t>(bkt) * L2 + i) * 32 + o] =
                    n.l2w[static_cast<std::size_t>(i) * (OB * 32) + bkt * 32 + o];
    // Float-net L1 the same way (the AVX2 input-major L1 reads the 32 weights of one input: contiguous
    // per bucket instead of rows at a 1 KB stride; same values, same fma order).
    if ((SC_FX_AVX2 || SC_FX_L1BM_NEON) && !n.quant && L2 == 32) {
        n.l1w_bm.resize(static_cast<std::size_t>(OB) * HL * 32);
        for (std::uint32_t bkt = 0; bkt < OB; ++bkt)
            for (std::uint32_t i = 0; i < HL; ++i)
                for (std::uint32_t o = 0; o < 32; ++o)
                    n.l1w_bm[(static_cast<std::size_t>(bkt) * HL + i) * 32 + o] =
                        n.l1w[static_cast<std::size_t>(i) * (OB * L2) + bkt * L2 + o];
    }
    build_tables();
    if (n.quant) permute_neurons(n);
#if SC_NNUE_PROBES
    g_base_only = std::getenv("SCNNUE_BASEONLY") != nullptr;
    g_verify = std::getenv("SCNNUE_VERIFY") != nullptr;
    g_no_finny = std::getenv("SCNNUE_NOFINNY") != nullptr;
    g_prefetch = std::getenv("SCNNUE_NOPREFETCH") == nullptr;
    g_no_cols = std::getenv("SCNNUE_NOCOLS") != nullptr;
    { const char* q = std::getenv("SCNNUE_SQUEEZE"); g_squeeze = q ? std::atoi(q) : 0; }
    g_no_update = std::getenv("SCNNUE_NOUPDATE") != nullptr;
    g_l1dense = std::getenv("SCNNUE_L1DENSE") != nullptr;
    if (g_verify) std::fprintf(stderr, "[nnue] SCNNUE_VERIFY on\n");   // verify_acc.py requires this line
#endif
    n.loaded = true;
    g_net = std::move(n);
    ++g_net_gen;
    eval_cache_clear();
    return true;
}

bool load(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    return load_stream(f);
}

// Same loader over an in-memory image of a net file (the embedded net, see embed_net.hpp): a
// read-only streambuf over the bytes, so the parser is shared and the two paths cannot diverge.
bool load_memory(const void* data, std::size_t size) {
    if (data == nullptr || size == 0) return false;
    struct membuf : std::streambuf {
        membuf(const char* b, std::size_t n) {
            char* c = const_cast<char*>(b);
            setg(c, c, c + n);
        }
    } buf(static_cast<const char*>(data), size);
    std::istream f(&buf);
    return load_stream(f);
}

bool loaded() noexcept { return g_net.loaded; }

// ---- S1: neuron permutation + sparse-L1 weight layout ---------------------------
// Reorders the hidden neurons by how often their pairwise activation is zero (most often first),
// measured over deterministic random games from the start position (160 plies, so later phases
// count too), then lays L1 out by 4-input
// chunk. A neuron j moves as a unit: its two accumulator halves (j, j + hl/2), its FT bias, its FT
// weight in every feature row, and its L1 weight in both the side-to-move and the other block.
// Every L1 sum is the same integer in any neuron order, so the evaluation is unchanged.
void permute_neurons(Net& n) {
    const int hl = n.hl, half = hl / 2, L2 = n.l2, OB = n.out_buckets;
    const std::size_t TI = n.total_inputs;
    const std::int8_t* l0 = n.l0w_i8.data();

    std::vector<std::uint32_t> zeros(static_cast<std::size_t>(half), 0);
    std::vector<std::int16_t> a_stm(static_cast<std::size_t>(hl)), a_ntm(static_cast<std::size_t>(hl));
    std::uint64_t rng = 0x9E3779B97F4A7C15ull;
    auto next = [&] { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; };
    auto cr = [](int x) { return x < 0 ? 0 : (x > QA ? QA : x); };
    for (int game = 0; game < 128; ++game) {
        Board b;
        for (int ply = 0; ply < 160; ++ply) {
            if (ply >= 6) {
                for (int j = 0; j < hl; ++j) a_stm[j] = a_ntm[j] = n.l0b_i[j];
                gather(
                    b, n.base_dims,
                    [&](std::size_t f) { const std::int8_t* w = l0 + f * hl; for (int j = 0; j < hl; ++j) a_stm[j] += w[j]; },
                    [&](std::size_t f) { const std::int8_t* w = l0 + f * hl; for (int j = 0; j < hl; ++j) a_ntm[j] += w[j]; });
                for (int j = 0; j < half; ++j) {
                    zeros[j] += (cr(a_stm[j]) * cr(a_stm[j + half]) / QA == 0);
                    zeros[j] += (cr(a_ntm[j]) * cr(a_ntm[j + half]) / QA == 0);
                }
            }
            Movelist ml;
            chess::movegen::legalmoves(ml, b);
            if (ml.empty()) break;
            b.makeMove(ml[static_cast<int>(next() % static_cast<std::uint64_t>(ml.size()))]);
        }
    }
    std::vector<int> perm(static_cast<std::size_t>(half));
    for (int j = 0; j < half; ++j) perm[j] = j;
    std::stable_sort(perm.begin(), perm.end(), [&](int x, int y) { return zeros[x] > zeros[y]; });

    // FT weights (every feature row) and bias.
    std::vector<std::int8_t> row8(static_cast<std::size_t>(hl));
    for (std::size_t r = 0; r < TI; ++r) {
        std::int8_t* row = n.l0w_i8.data() + r * hl;
        for (int j = 0; j < half; ++j) { row8[j] = row[perm[j]]; row8[half + j] = row[half + perm[j]]; }
        std::memcpy(row, row8.data(), static_cast<std::size_t>(hl));
    }
    std::vector<std::int16_t> b16(n.l0b_i);
    for (int j = 0; j < half; ++j) { n.l0b_i[j] = b16[perm[j]]; n.l0b_i[half + j] = b16[half + perm[j]]; }
    // L1 inputs: [0, half) the side to move's neurons, [half, hl) the other side's.
    for (std::size_t r = 0; r < static_cast<std::size_t>(OB) * L2; ++r) {
        std::int8_t* row = n.l1w_dot.data() + r * hl;
        for (int j = 0; j < half; ++j) { row8[j] = row[perm[j]]; row8[half + j] = row[half + perm[j]]; }
        std::memcpy(row, row8.data(), static_cast<std::size_t>(hl));
    }
    // Sparse layout: [bucket][chunk k][output o][4 inputs 4k..4k+3].
    const int nch = hl / 4;
    n.l1w_sp.resize(static_cast<std::size_t>(OB) * nch * L2 * 4);
    for (int bkt = 0; bkt < OB; ++bkt)
        for (int k = 0; k < nch; ++k)
            for (int o = 0; o < L2; ++o)
                for (int t = 0; t < 4; ++t)
                    n.l1w_sp[((static_cast<std::size_t>(bkt) * nch + k) * L2 + o) * 4 + t] =
                        n.l1w_dot[(static_cast<std::size_t>(bkt) * L2 + o) * hl + 4 * k + t];
}

}  // namespace engine::nnue
