/* Copyright (c) 2026  DC-DeepComputing RVV optimization
 *
 * RISC-V Vector (RVV 1.0) support macros for PFFFT, 4-element float vectors.
 *
 * Design (see rvv-hb references/highway-simd.md):
 *  - v4sf is a GCC fixed-size vector of 4 floats. On a RVV target GCC maps
 *    this to a single vector register with vsetivli e32,mf2 (128-bit). This
 *    keeps the type union-compatible, which PFFFT's core relies on for lane
 *    access (((v4sf_union*)p)[i].f[k]) -- a sizeless __riscv_v_* type cannot
 *    be a union member, so intrinsics-typed v4sf is not usable here.
 *  - VLEN-agnostic: vector_size(16) asks for 4 x e32 lanes; on VLEN=256 this
 *    is LMUL=mf2. The algorithm hardcodes SIMD_SZ=4 so we do not attempt a
 *    full-width 8-lane mapping.
 *
 * IMPORTANT (alias safety): INTERLEAVE2/UNINTERLEAVE2 are called in-place in
 * the complex backward transform (in1==out1, in2==out2). Both results must be
 * computed into temporaries BEFORE either output is written -- this mirrors
 * the SSE1 backend's tmp__ pattern. __builtin_shuffle on in-place args would
 * otherwise clobber the second input.
 */

#ifndef PF_RVV_FLT_H
#define PF_RVV_FLT_H

/*
  RISC-V RVV support macros (4-lane float vector).
  Enable via -march=rv64gcv* (auto-detected), or force-disable with
  PFFFT_SIMD_DISABLE to fall back to the scalar backend.
*/
#if !defined(SIMD_SZ) && !defined(PFFFT_SIMD_DISABLE) && \
    defined(__riscv) && defined(__riscv_vector) && (__riscv_xlen == 64)
#pragma message( __FILE__ ": RISC-V RVV float macros are defined" )

typedef float v4sf __attribute__((vector_size(16)));
typedef int    vi4  __attribute__((vector_size(16)));

#  define SIMD_SZ 4

typedef union v4sf_union {
  v4sf  v;
  float f[SIMD_SZ];
} v4sf_union;

#  define VARCH "RVV"
#  define VREQUIRES_ALIGN 0  /* RVV unaligned loads/stores are fine */

/* arithmetic: GCC emits vfadd/vfsub/vfmul/vfmadd single instructions */
#  define VZERO()          ((v4sf){ 0.f, 0.f, 0.f, 0.f })
#  define VMUL(a,b)        ((a) * (b))
#  define VADD(a,b)        ((a) + (b))
#  define VSUB(a,b)        ((a) - (b))
#  define VMADD(a,b,c)     ((a) * (b) + (c))
#  define LD_PS1(p)        ((v4sf){ (p), (p), (p), (p) })

/* unaligned load via vector deref (GCC uses vle32.v, no align fault) */
#  define VLOAD_UNALIGNED(ptr)  (*((const v4sf*)(ptr)))
#  define VLOAD_ALIGNED(ptr)    (*((const v4sf*)(ptr)))
#  define VALIGNED(ptr)         ((((uintptr_t)(ptr)) & 0xF) == 0)

/* INTERLEAVE2: out1={a0,b0,a1,b1}, out2={a2,b2,a3,b3}
   Temporaries first: callers use this in-place on complex backward FFT. */
#  define INTERLEAVE2(in1, in2, out1, out2) do { \
    v4sf t1_ = __builtin_shuffle((in1),(in2),(vi4){ 0, 4, 1, 5 }); \
    v4sf t2_ = __builtin_shuffle((in1),(in2),(vi4){ 2, 6, 3, 7 }); \
    (out1) = t1_; (out2) = t2_; \
  } while (0)

/* UNINTERLEAVE2: out1={a0,a2,b0,b2}, out2={a1,a3,b1,b3} */
#  define UNINTERLEAVE2(in1, in2, out1, out2) do { \
    v4sf t1_ = __builtin_shuffle((in1),(in2),(vi4){ 0, 2, 4, 6 }); \
    v4sf t2_ = __builtin_shuffle((in1),(in2),(vi4){ 1, 3, 5, 7 }); \
    (out1) = t1_; (out2) = t2_; \
  } while (0)

/* VTRANSPOSE4: 4x4 matrix transpose (in-place), temporaries first */
#  define VTRANSPOSE4(x0,x1,x2,x3) do { \
    v4sf t0l_ = __builtin_shuffle((x0),(x2),(vi4){ 0, 4, 1, 5 }); \
    v4sf t0h_ = __builtin_shuffle((x0),(x2),(vi4){ 2, 6, 3, 7 }); \
    v4sf t1l_ = __builtin_shuffle((x1),(x3),(vi4){ 0, 4, 1, 5 }); \
    v4sf t1h_ = __builtin_shuffle((x1),(x3),(vi4){ 2, 6, 3, 7 }); \
    v4sf o0_ = __builtin_shuffle(t0l_,t1l_,(vi4){ 0, 4, 1, 5 }); \
    v4sf o1_ = __builtin_shuffle(t0l_,t1l_,(vi4){ 2, 6, 3, 7 }); \
    v4sf o2_ = __builtin_shuffle(t0h_,t1h_,(vi4){ 0, 4, 1, 5 }); \
    v4sf o3_ = __builtin_shuffle(t0h_,t1h_,(vi4){ 2, 6, 3, 7 }); \
    (x0) = o0_; (x1) = o1_; (x2) = o2_; (x3) = o3_; \
  } while (0)

/* VSWAPHL: {b_low, a_high} = {b0,b1,a2,a3} */
#  define VSWAPHL(a,b)  __builtin_shuffle((a),(b),(vi4){ 4, 5, 2, 3 })

/* reverse all floats: {a3,a2,a1,a0} */
#  define VREV_S(a)     __builtin_shuffle((a),(vi4){ 3, 2, 1, 0 })

/* reverse complex pairs: {a2,a3,a0,a1} */
#  define VREV_C(a)     __builtin_shuffle((a),(vi4){ 2, 3, 0, 1 })

#else
/* #pragma message( __FILE__ ": RISC-V RVV float macros are not defined" ) */
#endif

#endif /* PF_RVV_FLT_H */
