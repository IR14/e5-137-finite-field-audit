#include <algorithm>
#include <cstdint>
#include <cstring>

namespace {

constexpr int kModulus = 137;
constexpr int kAxes = 26;
constexpr int kSubparticles = 5;
constexpr int kSupervectorBytes = kAxes * kSubparticles;
constexpr int kReferenceTokens = 120000;
constexpr int kReferenceVectors = 117;
constexpr int kWindowTokens = 1500;
constexpr std::uint8_t kErased = 255;

int mod137(int value) {
    int result = value % kModulus;
    return result < 0 ? result + kModulus : result;
}

int vector_count_for_tokens(int token_count) {
    if (token_count <= 0) {
        return 0;
    }
    if (token_count >= kReferenceTokens) {
        return kReferenceVectors;
    }
    int windows = (token_count + kWindowTokens - 1) / kWindowTokens;
    return std::max(1, std::min(kReferenceVectors, windows));
}

std::uint8_t axis_value(int core, int vector_index, int axis, int subparticle) {
    return static_cast<std::uint8_t>(mod137(core + axis * (subparticle + 1) + vector_index));
}

}  // namespace

extern "C" {

int e5137_compression_axis_count() {
    return kAxes;
}

int e5137_compression_subparticle_count() {
    return kSubparticles;
}

int e5137_compression_supervector_bytes() {
    return kSupervectorBytes;
}

int e5137_compression_window_tokens() {
    return kWindowTokens;
}

int e5137_compression_reference_vectors() {
    return kReferenceVectors;
}

int e5137_compression_vector_count_for_tokens(int token_count) {
    return vector_count_for_tokens(token_count);
}

int e5137_compression_archive_bytes_for_tokens(int token_count) {
    return vector_count_for_tokens(token_count) * kSupervectorBytes;
}

int e5137_compress_text_gf137(
    const std::uint8_t* text,
    int byte_count,
    int token_count,
    std::uint8_t* out_archive
) {
    if (text == nullptr || out_archive == nullptr || byte_count <= 0 || token_count <= 0) {
        return 0;
    }

    const int vector_count = vector_count_for_tokens(token_count);
    if (vector_count <= 0) {
        return 0;
    }

    for (int v = 0; v < vector_count; ++v) {
        std::uint32_t state[kSubparticles] = {
            static_cast<std::uint32_t>(mod137(42 + v)),
            static_cast<std::uint32_t>(mod137(69 + 2 * v)),
            static_cast<std::uint32_t>(mod137(117 + 3 * v)),
            static_cast<std::uint32_t>(mod137(26 + 5 * v)),
            static_cast<std::uint32_t>(mod137(5 + 7 * v)),
        };

        const int start = static_cast<int>((static_cast<long long>(v) * byte_count) / vector_count);
        const int end = static_cast<int>((static_cast<long long>(v + 1) * byte_count) / vector_count);
        for (int i = start; i < end; ++i) {
            const std::uint32_t b = static_cast<std::uint32_t>(text[i]);
            const std::uint32_t local_pos = static_cast<std::uint32_t>(i - start + 1);
            for (int s = 0; s < kSubparticles; ++s) {
                const std::uint32_t multiplier = static_cast<std::uint32_t>(17 + 2 * s);
                const std::uint32_t injection =
                    b + local_pos * static_cast<std::uint32_t>(s + 1) +
                    static_cast<std::uint32_t>((v + 1) * (s + 3));
                state[s] = (state[s] * multiplier + injection) % kModulus;
            }
        }

        for (int axis = 0; axis < kAxes; ++axis) {
            for (int s = 0; s < kSubparticles; ++s) {
                out_archive[v * kSupervectorBytes + axis * kSubparticles + s] =
                    axis_value(static_cast<int>(state[s]), v, axis, s);
            }
        }
    }

    return vector_count;
}

int e5137_repair_archive_gf137(
    const std::uint8_t* corrupted_archive,
    int vector_count,
    std::uint8_t* repaired_archive,
    int* min_votes_out
) {
    if (
        corrupted_archive == nullptr || repaired_archive == nullptr ||
        vector_count <= 0 || vector_count > kReferenceVectors
    ) {
        if (min_votes_out != nullptr) {
            *min_votes_out = 0;
        }
        return 0;
    }

    int global_min_votes = kAxes;
    for (int v = 0; v < vector_count; ++v) {
        int core[kSubparticles] = {0, 0, 0, 0, 0};
        for (int s = 0; s < kSubparticles; ++s) {
            int counts[kModulus];
            std::memset(counts, 0, sizeof(counts));
            for (int axis = 0; axis < kAxes; ++axis) {
                const std::uint8_t value =
                    corrupted_archive[v * kSupervectorBytes + axis * kSubparticles + s];
                if (value == kErased || value >= kModulus) {
                    continue;
                }
                const int candidate = mod137(static_cast<int>(value) - axis * (s + 1) - v);
                counts[candidate] += 1;
            }
            int best_value = 0;
            int best_votes = -1;
            for (int r = 0; r < kModulus; ++r) {
                if (counts[r] > best_votes) {
                    best_votes = counts[r];
                    best_value = r;
                }
            }
            core[s] = best_value;
            global_min_votes = std::min(global_min_votes, best_votes);
        }

        for (int axis = 0; axis < kAxes; ++axis) {
            for (int s = 0; s < kSubparticles; ++s) {
                repaired_archive[v * kSupervectorBytes + axis * kSubparticles + s] =
                    axis_value(core[s], v, axis, s);
            }
        }
    }

    if (min_votes_out != nullptr) {
        *min_votes_out = global_min_votes;
    }
    return global_min_votes >= 16 ? 1 : 0;
}

}  // extern "C"
