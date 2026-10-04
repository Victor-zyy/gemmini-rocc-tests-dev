#ifndef GEMMINI_LAYER_MATMUL_H
#define GEMMINI_LAYER_MATMUL_H
#include "gemmini.h"

/* ABI v1, funcs 30..35; requires a bitstream with HAS_LAYER_MATMUL.
 * Inputs are NHWC A and output-major B; B is transposed by the WS engine.
 * Keep K whole. Tile sizes are selected once, not in the CPU tile loop.
 */
static inline int gemmini_layer_plan(size_t m, size_t n, size_t k,
                                    size_t *ti, size_t *tj)
{
    if (!m || !n || !k || m > UINT32_MAX || n > UINT16_MAX || k > UINT16_MAX)
        return -1;
    size_t kb = (k + DIM - 1) / DIM;
    size_t jb = (n + DIM - 1) / DIM;
    if (jb > 4) jb = 4;
    for (; jb; --jb) {
        size_t ib = (m + DIM - 1) / DIM;
        size_t acc_limit = (ACC_ROWS / 2) / (jb * DIM);
        size_t sum_limit = (BANK_NUM * BANK_ROWS / 2) / (kb * DIM);
        if (sum_limit <= jb) continue;
        if (ib > acc_limit) ib = acc_limit;
        if (ib > sum_limit - jb) ib = sum_limit - jb;
        if (ib > UINT16_MAX / DIM) ib = UINT16_MAX / DIM;
        if (ib) { *ti = ib * DIM; *tj = jb * DIM; return 0; }
    }
    return -1;
}

static inline int gemmini_layer_matmul(size_t m, size_t n, size_t k,
        const elem_t *a, const elem_t *b, const acc_t *d, void *c,
        size_t sa, size_t sb, size_t sc, bool full, int act, acc_scale_t scale)
{
    size_t ti, tj;
    if (gemmini_layer_plan(m, n, k, &ti, &tj) || !a || !b || !c ||
        sa < k || sb < k || sc < n || sa > UINT32_MAX || sb > UINT32_MAX ||
        sc > UINT32_MAX / (full ? sizeof(acc_t) : sizeof(elem_t)) ||
        (act != NO_ACTIVATION && act != SILU) || (full && act != NO_ACTIVATION))
        return -1;
    __asm__ volatile("" ::: "memory");
    gemmini_extended_config_ex(WS, 0, 0, 1, false, true);
    gemmini_extended_config_st(sc * (full ? sizeof(acc_t) : sizeof(elem_t)), act & 3, scale);
    gemmini_extended3_config_ld(sa, MVIN_SCALE_IDENTITY, false, 0);
    gemmini_extended3_config_ld(sb, MVIN_SCALE_IDENTITY, false, 1);
    gemmini_extended3_config_ld(0, MVIN_SCALE_IDENTITY, false, 2);
    if (act == SILU) gemmini_config_norm(0, 0, 0, 1, 0, 0, 0);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, m, ((uint64_t)k << 32) | n, 30);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ti, tj, 31);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, a, b, 32);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, d, c, 33);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, sa, ((uint64_t)sc << 32) | sb, 34);
    ROCC_INSTRUCTION_RS1_RS2(XCUSTOM_ACC, ((uint64_t)act << 8) |
        ((uint64_t)full << 1) | (d != NULL), 0, 35);
    gemmini_fence();
    __asm__ volatile("" ::: "memory");
    return 0;
}
#endif
