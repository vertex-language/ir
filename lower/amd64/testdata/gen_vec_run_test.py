# Generates ir/lower/amd64/vec_run_test.go: every §V verb against a scalar C
# reference of its definition, and every §V5 and §A8 verb x86-64 has an
# instruction for against a scalar C reference of the IR's definition, built
# for x86-64 and run. Usage: python3 gen_vec_run_test.py ../vec_run_test.go
#
# The §V list is arm64's (testdata/gen_vec_run_test.py there), so that both
# backends answer to the same definitions. The verbs whose instruction is
# past SSE2 run twice: once in a function with nothing but the baseline,
# where ExpandVectorVerbs has rewritten the byte permutes, and once in one
# that requires aes, which brings SSSE3 and PSHUFB with it.
#
# The cryptographic references are scalar rather than the instructions'
# intrinsics: SHA-NI keeps its state in another lane order and adds its own
# constants, and the lowering's job is to translate, which an intrinsic of
# the same instruction would not check. A case the processor cannot run --
# SHA under Rosetta, which has AES-NI, PCLMULQDQ and SSE4.2 but not SHA --
# prints SKIP and the test logs it rather than failing. On an Apple host, SHA
# is emulated instead: see sha_emulate in the harness.
import sys
OUT = sys.argv[1]
T = []  # (name, go body, C reference, kind, caps)
def add(name, go, cref, kind="v", caps=None):
    T.append((name, go, cref, kind, caps))

B = {"i8x16":(16,"8"),"i16x8":(8,"16"),"i32x4":(4,"32"),"i64x2":(2,"64")}
def bin_lane(shape, op, ctype):
    n, w = B[shape]
    return "for (int i = 0; i < %d; i++) { %s x = G%s(a,i), y = G%s(b,i); P%s(r,i,(%s)(%s)); }" % (n, ctype, ctype.upper()[:1]+w if False else ctype, ctype, w, ctype, op)

# typed lane accessors in C: get_u8..get_i64 and put
def L(shape, signed):
    n, w = B[shape]
    t = ("int%s_t" if signed else "uint%s_t") % w
    return n, w, t

def lanewise(shape, signed, expr):
    n, w, t = L(shape, signed)
    return ("for (int i = 0; i < %d; i++) { %s x = get%s%s(a, i), y = get%s%s(b, i); put%s(r, i, (uint%s_t)(%s)); }"
            % (n, t, "s" if signed else "u", w, "s" if signed else "u", w, w, w, expr))

add("and", "v.And(a, b)", "for (int i=0;i<16;i++) r[i]=a[i]&b[i];")
add("or", "v.Or(a, b)", "for (int i=0;i<16;i++) r[i]=a[i]|b[i];")
add("xor", "v.Xor(a, b)", "for (int i=0;i<16;i++) r[i]=a[i]^b[i];")
add("not", "v.Not(a)", "for (int i=0;i<16;i++) r[i]=(uint8_t)~a[i];")
add("andnot", "v.AndNot(a, b)", "for (int i=0;i<16;i++) r[i]=a[i]&(uint8_t)~b[i];")
add("xor3", "v.Xor3(a, b, c)", "for (int i=0;i<16;i++) r[i]=a[i]^b[i]^c[i];")
add("xor_andnot", "v.XorAndNot(a, b, c)", "for (int i=0;i<16;i++) r[i]=a[i]^(b[i]&(uint8_t)~c[i]);")
for k in (0,3,8,15,16):
    add("shl_bytes_%d"%k, "v.ShlBytes(a, %d)"%k, "for (int i=0;i<16;i++) r[i] = i >= %d ? a[i-%d] : 0;"%(k,k))
    add("shr_bytes_%d"%k, "v.ShrBytes(a, %d)"%k, "for (int i=0;i<16;i++) r[i] = i + %d < 16 ? a[i+%d] : 0;"%(k,k))
add("zext_i32", "v.ZExtI32(n)", "memset(r,0,16); putu32(r,0,(uint32_t)n);")
add("zext_i64", "v.ZExtI64(x64)", "memset(r,0,16); putu64(r,0,getu64(a,0));")
add("const", "v.Const([16]byte{1, 0x80, 0xff, 0, 7, 6, 5, 4, 3, 2, 1, 0, 0xaa, 0x55, 0, 9})",
    "static const uint8_t k[16]={1,0x80,0xff,0,7,6,5,4,3,2,1,0,0xaa,0x55,0,9}; memcpy(r,k,16);")
add("const_zero", "v.Zero()", "memset(r,0,16);")
add("const_mask", "v.Const([16]byte{0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff, 0xff, 0, 0xff, 0, 0, 0, 0xff, 0xff})",
    "static const uint8_t k[16]={0xff,0,0xff,0,0,0,0xff,0xff,0xff,0,0xff,0,0,0,0xff,0xff}; memcpy(r,k,16);")
add("select", "v.Select(e.I32.Ne(e.I32.And(n, e.I32.Const(1)), e.I32.Const(0)), a, b)", "memcpy(r, (n & 1) ? a : b, 16);")
for sh in B:
    nn, w = B[sh]
    add(sh+"_add", "v.%sAdd(a, b)" % sh.capitalize().replace("x","x"), lanewise(sh, False, "x + y"))
    add(sh+"_sub", "v.%sSub(a, b)" % sh.capitalize(), lanewise(sh, False, "x - y"))
add("i16x8_mul", "v.I16x8Mul(a, b)", lanewise("i16x8", False, "x * y"))
add("i16x8_mulhi_s", "v.I16x8MulHiS(a, b)", lanewise("i16x8", True, "((int32_t)x * y) >> 16"))
add("i16x8_mulhi_u", "v.I16x8MulHiU(a, b)", lanewise("i16x8", False, "((uint32_t)x * y) >> 16"))
add("i32x4_mul_even_u", "v.I32x4MulEvenU(a, b)", "putu64(r,0,(uint64_t)getu32(a,0)*getu32(b,0)); putu64(r,1,(uint64_t)getu32(a,2)*getu32(b,2));")
def sat(sh, signed, op):
    n, w, t = L(sh, signed)
    lo, hi = ("(-(1<<(%s-1)))" % w, "((1<<(%s-1))-1)" % w) if signed else ("0", "((1<<%s)-1)" % w)
    return lanewise(sh, signed, "clampi((int)x %s (int)y, %s, %s)" % (op, lo, hi))
for sh in ("i8x16","i16x8"):
    S = sh.capitalize()
    add(sh+"_add_sat_s", "v.%sAddSatS(a, b)"%S, sat(sh, True, "+"))
    add(sh+"_add_sat_u", "v.%sAddSatU(a, b)"%S, sat(sh, False, "+"))
    add(sh+"_sub_sat_s", "v.%sSubSatS(a, b)"%S, sat(sh, True, "-"))
    add(sh+"_sub_sat_u", "v.%sSubSatU(a, b)"%S, sat(sh, False, "-"))
    add(sh+"_avgr_u", "v.%sAvgrU(a, b)"%S, lanewise(sh, False, "((uint32_t)x + y + 1) >> 1"))
add("i8x16_min_u", "v.I8x16MinU(a, b)", lanewise("i8x16", False, "x < y ? x : y"))
add("i8x16_max_u", "v.I8x16MaxU(a, b)", lanewise("i8x16", False, "x > y ? x : y"))
add("i16x8_min_s", "v.I16x8MinS(a, b)", lanewise("i16x8", True, "x < y ? x : y"))
add("i16x8_max_s", "v.I16x8MaxS(a, b)", lanewise("i16x8", True, "x > y ? x : y"))
add("i8x16_sad_u", "v.I8x16SadU(a, b)", "for (int h=0;h<2;h++){ uint64_t s=0; for(int i=0;i<8;i++){int d=(int)a[8*h+i]-(int)b[8*h+i]; s+= d<0?-d:d;} putu64(r,h,s);} ")
add("i16x8_madd_s", "v.I16x8MaddS(a, b)", "for (int k=0;k<4;k++) putu32(r,k,(uint32_t)((int32_t)gets16(a,2*k)*gets16(b,2*k) + (int32_t)gets16(a,2*k+1)*gets16(b,2*k+1)));")
for sh in ("i8x16","i16x8","i32x4"):
    S = sh.capitalize()
    add(sh+"_eq", "v.%sEq(a, b)"%S, lanewise(sh, False, "x == y ? -1 : 0"))
    add(sh+"_gt_s", "v.%sGtS(a, b)"%S, lanewise(sh, True, "x > y ? -1 : 0"))
def shiftref(sh, kind):
    n, w = B[sh]
    if kind == "shl":
        return "for (int i=0;i<%d;i++){ uint%s_t x=getu%s(a,i); putu%s(r,i,(uint%s_t)((uint32_t)n >= %s ? 0 : (uint64_t)x << n)); }" % (n,w,w,w,w,w)
    if kind == "shr_u":
        return "for (int i=0;i<%d;i++){ uint%s_t x=getu%s(a,i); putu%s(r,i,(uint%s_t)((uint32_t)n >= %s ? 0 : x >> n)); }" % (n,w,w,w,w,w)
    return "for (int i=0;i<%d;i++){ int%s_t x=gets%s(a,i); putu%s(r,i,(uint%s_t)(x >> ((uint32_t)n >= %s ? %s-1 : n))); }" % (n,w,w,w,w,w,w)
for sh in ("i16x8","i32x4","i64x2"):
    S = sh.capitalize()
    add(sh+"_shl", "v.%sShl(a, n)"%S, shiftref(sh,"shl"))
    add(sh+"_shr_u", "v.%sShrU(a, n)"%S, shiftref(sh,"shr_u"))
    add(sh+"_shl_k5", "v.%sShl(a, e.I32.Const(5))"%S, shiftref(sh,"shl").replace("n","5").replace("i<5","i<n") if False else shiftref(sh,"shl").replace("(uint32_t)n","5u").replace("<< n","<< 5"))
for sh in ("i16x8","i32x4"):
    S = sh.capitalize()
    add(sh+"_shr_s", "v.%sShrS(a, n)"%S, shiftref(sh,"shr_s"))
add("i8x16_bitmask", "v.I8x16Bitmask(a)", "uint32_t m=0; for(int i=0;i<16;i++) m |= (uint32_t)(a[i]>>7) << i; putu32(r,0,m);", kind="w")
add("i8x16_splat", "v.I8x16Splat(n)", "for(int i=0;i<16;i++) r[i]=(uint8_t)n;")
add("i16x8_splat", "v.I16x8Splat(n)", "for(int i=0;i<8;i++) putu16(r,i,(uint16_t)n);")
add("i32x4_splat", "v.I32x4Splat(n)", "for(int i=0;i<4;i++) putu32(r,i,(uint32_t)n);")
add("i64x2_splat", "v.I64x2Splat(x64)", "for(int i=0;i<2;i++) putu64(r,i,getu64(a,0));")
add("i16x8_extract_lane_u", "v.I16x8ExtractLaneU(a, 5)", "putu32(r,0,getu16(a,5));", kind="w")
add("i32x4_extract_lane", "v.I32x4ExtractLane(a, 3)", "putu32(r,0,getu32(a,3));", kind="w")
add("i64x2_extract_lane", "v.I64x2ExtractLane(a, 1)", "putu64(r,0,getu64(a,1));", kind="x")
add("i16x8_replace_lane", "v.I16x8ReplaceLane(a, n, 6)", "memcpy(r,a,16); putu16(r,6,(uint16_t)n);")
for p in (0x1b, 0xe4, 0x00, 0x8d):
    add("i32x4_shuffle_%x"%p, "v.I32x4Shuffle(a, 0x%x)"%p, "for(int l=0;l<4;l++) putu32(r,l,getu32(a,(0x%x>>(2*l))&3));"%p)
    add("i16x8_shuffle_low_%x"%p, "v.I16x8ShuffleLow(a, 0x%x)"%p, "memcpy(r,a,16); for(int l=0;l<4;l++) putu16(r,l,getu16(a,(0x%x>>(2*l))&3));"%p)
    add("i16x8_shuffle_high_%x"%p, "v.I16x8ShuffleHigh(a, 0x%x)"%p, "memcpy(r,a,16); for(int l=0;l<4;l++) putu16(r,4+l,getu16(a,4+((0x%x>>(2*l))&3)));"%p)
for sh in B:
    nn, w = B[sh]
    S = sh.capitalize()
    half = nn // 2
    add(sh+"_unpack_low", "v.%sUnpackLow(a, b)"%S, "for(int i=0;i<%d;i++){ putu%s(r,2*i,getu%s(a,i)); putu%s(r,2*i+1,getu%s(b,i)); }"%(half,w,w,w,w))
    add(sh+"_unpack_high", "v.%sUnpackHigh(a, b)"%S, "for(int i=0;i<%d;i++){ putu%s(r,2*i,getu%s(a,%d+i)); putu%s(r,2*i+1,getu%s(b,%d+i)); }"%(half,w,w,half,w,w,half))
add("i8x16_narrow_s", "v.I8x16NarrowS(a, b)", "for(int i=0;i<8;i++){ r[i]=(uint8_t)clampi(gets16(a,i),-128,127); r[8+i]=(uint8_t)clampi(gets16(b,i),-128,127);}")
add("i8x16_narrow_u", "v.I8x16NarrowU(a, b)", "for(int i=0;i<8;i++){ r[i]=(uint8_t)clampi(gets16(a,i),0,255); r[8+i]=(uint8_t)clampi(gets16(b,i),0,255);}")
add("i16x8_narrow_s", "v.I16x8NarrowS(a, b)", "for(int i=0;i<4;i++){ putu16(r,i,(uint16_t)clampi(gets32(a,i),-32768,32767)); putu16(r,4+i,(uint16_t)clampi(gets32(b,i),-32768,32767));}")
add("i8x16_swizzle", "v.I8x16Swizzle(a, b)", "for(int i=0;i<16;i++) r[i] = (b[i] & 0x1f) < 16 && b[i] < 16 ? a[b[i]] : 0;")
add("i16x8_bswap", "v.I16x8Bswap(a)", "for(int i=0;i<8;i++) putu16(r,i,__builtin_bswap16(getu16(a,i)));")
add("i32x4_bswap", "v.I32x4Bswap(a)", "for(int i=0;i<4;i++) putu32(r,i,__builtin_bswap32(getu32(a,i)));")
add("i64x2_bswap", "v.I64x2Bswap(a)", "for(int i=0;i<2;i++) putu64(r,i,__builtin_bswap64(getu64(a,i)));")
for k in (1,7,8,12,16,25):
    add("i32x4_rotl_%d"%k, "v.I32x4Rotl(a, %d)"%k, "for(int i=0;i<4;i++){uint32_t x=getu32(a,i); putu32(r,i,(x<<%d)|(x>>(32-%d)));}"%(k,k))
for k in (1,13,32,63):
    add("i64x2_rotl_%d"%k, "v.I64x2Rotl(a, %d)"%k, "for(int i=0;i<2;i++){uint64_t x=getu64(a,i); putu64(r,i,(x<<%d)|(x>>(64-%d)));}"%(k,k))
# The PSHUFB and SSSE3 paths of the verbs above whose lowering depends on it.
for name, gobody, cref, kind, caps in list(T):
    if name in ("i8x16_swizzle", "i16x8_bswap", "i32x4_bswap", "i64x2_bswap") or "_rotl_" in name:
        add(name + "_ssse3", gobody, cref, kind, "ir.CapAES")

# §V5 and §A8, against scalar references of the IR's definitions.
add("aes_enc", "v.AesEnc(a, b)", "aesround(r, a, b, 1, 1);", caps="ir.CapAES")
add("aes_enc_last", "v.AesEncLast(a, b)", "aesround(r, a, b, 1, 0);", caps="ir.CapAES")
add("aes_dec", "v.AesDec(a, b)", "aesround(r, a, b, 0, 1);", caps="ir.CapAES")
add("aes_dec_last", "v.AesDecLast(a, b)", "aesround(r, a, b, 0, 0);", caps="ir.CapAES")
add("aes_inv_mix_columns", "v.AesInvMixColumns(a)", "memcpy(r, a, 16); mixcolumns(r, 0);", caps="ir.CapAES")
for rc in (0, 1, 0x1b, 0x80):
    add("aes_keygen_assist_%x" % rc, "v.AesKeygenAssist(a, 0x%x)" % rc, "keygen(r, a, 0x%x);" % rc, caps="ir.CapAES")
for sel in range(4):
    add("i64x2_clmul_%d" % sel, "v.I64x2Clmul(a, b, %d)" % sel, "clmul(r, a, b, %d);" % sel, caps="ir.CapClmul")
add("sha1_rounds_ch", "v.Sha1RoundsCh(a, n, b)", "sha1rounds(r, a, (uint32_t)n, b, 0);", caps="ir.CapSHA1")
add("sha1_rounds_parity", "v.Sha1RoundsParity(a, n, b)", "sha1rounds(r, a, (uint32_t)n, b, 1);", caps="ir.CapSHA1")
add("sha1_rounds_maj", "v.Sha1RoundsMaj(a, n, b)", "sha1rounds(r, a, (uint32_t)n, b, 2);", caps="ir.CapSHA1")
add("sha1_msg1", "v.Sha1Msg1(a, b, c)", "for (int i = 0; i < 4; i++) putu32(r, i, getu32(a, i) ^ (i < 2 ? getu32(a, i + 2) : getu32(b, i - 2)) ^ getu32(c, i));", caps="ir.CapSHA1")
add("sha1_msg2", "v.Sha1Msg2(a, b)", "sha1msg2(r, a, b);", caps="ir.CapSHA1")
add("sha256_rounds_abcd", "v.Sha256RoundsAbcd(a, b, c)", "uint8_t e[16]; sha256rounds(r, e, a, b, c);", caps="ir.CapSHA256")
add("sha256_rounds_efgh", "v.Sha256RoundsEfgh(a, b, c)", "uint8_t d[16]; sha256rounds(d, r, b, a, c);", caps="ir.CapSHA256")
# Both halves in one function, which is the pair the lowering computes once.
add("sha256_rounds_pair", "v.I32x4Add(v.Sha256RoundsAbcd(a, b, c), v.I64x2Add(v.Sha256RoundsEfgh(b, a, c), v.Sha256RoundsAbcd(a, b, c)))",
    "uint8_t x[16], y[16], t[16]; sha256rounds(x, y, a, b, c); for (int i = 0; i < 2; i++) putu64(t, i, getu64(y, i) + getu64(x, i)); for (int i = 0; i < 4; i++) putu32(r, i, getu32(x, i) + getu32(t, i));",
    caps="ir.CapSHA256")
add("sha256_msg1", "v.Sha256Msg1(a, b)", "for (int i = 0; i < 4; i++) putu32(r, i, getu32(a, i) + s0(i < 3 ? getu32(a, i + 1) : getu32(b, 0)));", caps="ir.CapSHA256")
add("sha256_msg2", "v.Sha256Msg2(a, b, c)", "sha256msg2(r, a, b, c);", caps="ir.CapSHA256")
for verb, bits, data in [("Crc32cU8", 8, "e.I32.Load(pa)"), ("Crc32cU16", 16, "e.I32.Load(pa)"), ("Crc32cU32", 32, "e.I32.Load(pa)"), ("Crc32cU64", 64, "x64")]:
    add(verb.lower(), "e.I32.%s(n, %s)" % (verb, data), "putu32(r, 0, crc32c((uint32_t)n, getu64(a, 0), %d));" % bits, kind="w", caps="ir.CapCRC32C")


HARNESS = r"""const vecHarnessHead = `#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <cpuid.h>
#define G(T, W) static T get##W(const uint8_t* p, int i) { T v; memcpy(&v, p + i * sizeof(T), sizeof(T)); return v; }
G(uint16_t, u16) G(uint32_t, u32) G(uint64_t, u64) G(int16_t, s16) G(int32_t, s32) G(int64_t, s64)
static uint8_t getu8(const uint8_t* p, int i) { return p[i]; }
static int8_t gets8(const uint8_t* p, int i) { return (int8_t)p[i]; }
#define P(T, W) static void put##W(uint8_t* p, int i, T v) { memcpy(p + i * sizeof(T), &v, sizeof(T)); }
P(uint8_t, 8) P(uint16_t, 16) P(uint32_t, 32) P(uint64_t, 64)
static void putu8(uint8_t* p, int i, uint8_t v) { p[i] = v; }
static void putu16(uint8_t* p, int i, uint16_t v) { put16(p, i, v); }
static void putu32(uint8_t* p, int i, uint32_t v) { put32(p, i, v); }
static void putu64(uint8_t* p, int i, uint64_t v) { put64(p, i, v); }
static int clampi(int x, int lo, int hi) { return x < lo ? lo : x > hi ? hi : x; }

/* What a baseline function's swizzle calls, as the runtime defines it. */
void swizzle(uint8_t* r, const uint8_t* t, const uint8_t* i) __asm__("vertex_i8x16_swizzle");
void swizzle(uint8_t* r, const uint8_t* t, const uint8_t* i) { for (int k = 0; k < 16; k++) r[k] = i[k] < 16 ? t[i[k]] : 0; }

#define HAS_AES 1
#define HAS_CLMUL 2
#define HAS_SHA 4
#define HAS_SSE42 8
static int features(void) {
    unsigned a, b, c, d, f = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        if (c & (1u << 25)) f |= HAS_AES;
        if (c & (1u << 1)) f |= HAS_CLMUL;
        if (c & (1u << 20)) f |= HAS_SSE42;
    }
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d) && (b & (1u << 29))) f |= HAS_SHA;
    return f;
}

/* FIPS-197, byte 4c + r of the state being row r of column c. */
static uint8_t sbox[256], inv[256];
static uint8_t xt(uint8_t x) { return (uint8_t)((x << 1) ^ ((x >> 7) * 0x1b)); }
static uint8_t mul(uint8_t x, uint8_t y) { uint8_t p = 0; while (y) { if (y & 1) p ^= x; x = xt(x); y >>= 1; } return p; }
static void boxes(void) {
    for (int i = 0; i < 256; i++) {
        uint8_t x = 0;
        for (int j = 1; j < 256 && i; j++) if (mul((uint8_t)i, (uint8_t)j) == 1) { x = (uint8_t)j; break; }
        uint8_t s = x ^ (uint8_t)((x << 1) | (x >> 7)) ^ (uint8_t)((x << 2) | (x >> 6)) ^ (uint8_t)((x << 3) | (x >> 5)) ^ (uint8_t)((x << 4) | (x >> 4)) ^ 0x63;
        sbox[i] = s;
        inv[s] = (uint8_t)i;
    }
}
static void mixcolumns(uint8_t* s, int enc) {
    for (int c = 0; c < 4; c++) {
        uint8_t* p = s + 4 * c;
        uint8_t a0 = p[0], a1 = p[1], a2 = p[2], a3 = p[3];
        if (enc) {
            p[0] = mul(a0, 2) ^ mul(a1, 3) ^ a2 ^ a3; p[1] = a0 ^ mul(a1, 2) ^ mul(a2, 3) ^ a3;
            p[2] = a0 ^ a1 ^ mul(a2, 2) ^ mul(a3, 3); p[3] = mul(a0, 3) ^ a1 ^ a2 ^ mul(a3, 2);
        } else {
            p[0] = mul(a0, 14) ^ mul(a1, 11) ^ mul(a2, 13) ^ mul(a3, 9); p[1] = mul(a0, 9) ^ mul(a1, 14) ^ mul(a2, 11) ^ mul(a3, 13);
            p[2] = mul(a0, 13) ^ mul(a1, 9) ^ mul(a2, 14) ^ mul(a3, 11); p[3] = mul(a0, 11) ^ mul(a1, 13) ^ mul(a2, 9) ^ mul(a3, 14);
        }
    }
}
static void aesround(uint8_t* r, const uint8_t* a, const uint8_t* key, int enc, int mix) {
    uint8_t t[16];
    for (int c = 0; c < 4; c++)
        for (int w = 0; w < 4; w++)
            t[4 * c + w] = enc ? sbox[a[4 * ((c + w) % 4) + w]] : inv[a[4 * ((c - w + 4) % 4) + w]];
    if (mix) mixcolumns(t, enc);
    for (int i = 0; i < 16; i++) r[i] = t[i] ^ key[i];
}
static uint32_t subword(uint32_t w) {
    return (uint32_t)sbox[w & 0xff] | (uint32_t)sbox[(w >> 8) & 0xff] << 8 | (uint32_t)sbox[(w >> 16) & 0xff] << 16 | (uint32_t)sbox[w >> 24] << 24;
}
static void keygen(uint8_t* r, const uint8_t* a, int rcon) {
    uint32_t x1 = subword(getu32(a, 1)), x3 = subword(getu32(a, 3));
    putu32(r, 0, x1); putu32(r, 1, ((x1 >> 8) | (x1 << 24)) ^ (uint32_t)rcon);
    putu32(r, 2, x3); putu32(r, 3, ((x3 >> 8) | (x3 << 24)) ^ (uint32_t)rcon);
}
static void clmul(uint8_t* r, const uint8_t* a, const uint8_t* b, int sel) {
    uint64_t x = getu64(a, sel & 1), y = getu64(b, sel >> 1), lo = 0, hi = 0;
    for (int i = 0; i < 64; i++)
        if (y >> i & 1) { lo ^= x << i; if (i) hi ^= x >> (64 - i); }
    putu64(r, 0, lo); putu64(r, 1, hi);
}

/* SHA-1 and SHA-256 as the IR defines the verbs: Arm's instructions. */
static uint32_t rol(uint32_t x, int k) { return (x << k) | (x >> (32 - k)); }
static uint32_t ror(uint32_t x, int k) { return (x >> k) | (x << (32 - k)); }
static void sha1rounds(uint8_t* r, const uint8_t* abcd, uint32_t e, const uint8_t* wk, int f) {
    uint32_t x[4], y = e;
    for (int i = 0; i < 4; i++) x[i] = getu32(abcd, i);
    for (int i = 0; i < 4; i++) {
        uint32_t t = f == 0 ? (x[1] & x[2]) | (~x[1] & x[3]) : f == 1 ? x[1] ^ x[2] ^ x[3] : (x[1] & x[2]) | (x[1] & x[3]) | (x[2] & x[3]);
        y += rol(x[0], 5) + t + getu32(wk, i);
        x[1] = rol(x[1], 30);
        uint32_t top = x[3];
        x[3] = x[2]; x[2] = x[1]; x[1] = x[0]; x[0] = y; y = top;
    }
    for (int i = 0; i < 4; i++) putu32(r, i, x[i]);
}
static void sha1msg2(uint8_t* r, const uint8_t* x, const uint8_t* w12) {
    uint32_t w[4];
    for (int i = 0; i < 4; i++) {
        uint32_t t = getu32(x, i) ^ (i < 3 ? getu32(w12, i + 1) : w[0]);
        w[i] = rol(t, 1);
    }
    for (int i = 0; i < 4; i++) putu32(r, i, w[i]);
}
static uint32_t s0(uint32_t x) { return ror(x, 7) ^ ror(x, 18) ^ (x >> 3); }
static uint32_t s1(uint32_t x) { return ror(x, 17) ^ ror(x, 19) ^ (x >> 10); }
static void sha256rounds(uint8_t* abcdOut, uint8_t* efghOut, const uint8_t* abcd, const uint8_t* efgh, const uint8_t* wk) {
    uint32_t s[8];
    for (int i = 0; i < 4; i++) { s[i] = getu32(abcd, i); s[4 + i] = getu32(efgh, i); }
    for (int i = 0; i < 4; i++) {
        uint32_t a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
        uint32_t t1 = h + (ror(e, 6) ^ ror(e, 11) ^ ror(e, 25)) + ((e & f) ^ (~e & g)) + getu32(wk, i);
        uint32_t t2 = (ror(a, 2) ^ ror(a, 13) ^ ror(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
        s[7] = g; s[6] = f; s[5] = e; s[4] = d + t1; s[3] = c; s[2] = b; s[1] = a; s[0] = t1 + t2;
    }
    for (int i = 0; i < 4; i++) { putu32(abcdOut, i, s[i]); putu32(efghOut, i, s[4 + i]); }
}
static void sha256msg2(uint8_t* r, const uint8_t* x, const uint8_t* w8, const uint8_t* w12) {
    uint32_t w[6];
    w[0] = getu32(w12, 2); w[1] = getu32(w12, 3);
    for (int j = 0; j < 4; j++) {
        uint32_t w9 = j < 3 ? getu32(w8, j + 1) : getu32(w12, 0);
        w[2 + j] = getu32(x, j) + w9 + s1(w[j]);
        putu32(r, j, w[2 + j]);
    }
}
static uint32_t crc32c(uint32_t crc, uint64_t data, int bits) {
    if (bits < 64) data &= (1ull << bits) - 1;
    for (int i = 0; i < bits; i++) {
        uint32_t in = (crc ^ (uint32_t)(data >> i)) & 1;
        crc = (crc >> 1) ^ (in ? 0x82F63B78u : 0);
    }
    return crc;
}
#ifdef __APPLE__
/* SHA-NI where Rosetta has none: SIGILL, and the instruction emulated from
   Intel's pseudocode on the signal context's XMM registers. That is a
   second reading of SHA-NI, independent of the references above, which
   are Arm's; the lowering translates between the two, and this is how a
   machine without SHA checks the translation by running it. Register
   forms only, which is all the lowering emits. */
#include <signal.h>
#include <sys/ucontext.h>
static uint32_t ln(const uint8_t* x, int i) { return getu32(x, i); }
static void sha_emulate(int sig, siginfo_t* si, void* ctx) {
    (void)sig; (void)si;
    ucontext_t* uc = ctx;
    uint8_t* ip = (uint8_t*)uc->uc_mcontext->__ss.__rip;
    uint8_t* x0 = (uint8_t*)&uc->uc_mcontext->__fs.__fpu_xmm0;
    int len = 0, rex = 0;
    if ((ip[0] & 0xf0) == 0x40) { rex = ip[0]; len = 1; }
    if (ip[len] != 0x0f) { signal(SIGILL, SIG_DFL); return; }
    int map = ip[len + 1], op = ip[len + 2], modrm = ip[len + 3];
    len += 4;
    if ((modrm >> 6) != 3) { signal(SIGILL, SIG_DFL); return; }
    uint8_t* d = x0 + 16 * ((((rex >> 2) & 1) << 3) | ((modrm >> 3) & 7));
    uint8_t* s = x0 + 16 * (((rex & 1) << 3) | (modrm & 7));
    uint32_t r[4];
    if (map == 0x38 && op == 0xcb) { /* sha256rnds2 */
        uint32_t A = ln(s, 3), B = ln(s, 2), C = ln(d, 3), D = ln(d, 2), E = ln(s, 1), F = ln(s, 0), G = ln(d, 1), H = ln(d, 0);
        for (int i = 0; i < 2; i++) {
            uint32_t t = ((E & F) ^ (~E & G)) + (ror(E, 6) ^ ror(E, 11) ^ ror(E, 25)) + ln(x0, i) + H;
            uint32_t a = t + ((A & B) ^ (A & C) ^ (B & C)) + (ror(A, 2) ^ ror(A, 13) ^ ror(A, 22));
            uint32_t e = t + D;
            H = G; G = F; F = E; E = e; D = C; C = B; B = A; A = a;
        }
        r[3] = A; r[2] = B; r[1] = E; r[0] = F;
    } else if (map == 0x38 && op == 0xcc) { /* sha256msg1 */
        for (int i = 0; i < 4; i++) r[i] = ln(d, i) + s0(i < 3 ? ln(d, i + 1) : ln(s, 0));
    } else if (map == 0x38 && op == 0xcd) { /* sha256msg2 */
        uint32_t w[6] = {ln(s, 2), ln(s, 3)};
        for (int i = 0; i < 4; i++) w[2 + i] = ln(d, i) + s1(w[i]);
        for (int i = 0; i < 4; i++) r[i] = w[2 + i];
    } else if (map == 0x38 && op == 0xca) { /* sha1msg2 */
        uint32_t w16 = rol(ln(d, 3) ^ ln(s, 2), 1);
        r[3] = w16; r[2] = rol(ln(d, 2) ^ ln(s, 1), 1); r[1] = rol(ln(d, 1) ^ ln(s, 0), 1); r[0] = rol(ln(d, 0) ^ w16, 1);
    } else if (map == 0x3a && op == 0xcc) { /* sha1rnds4 */
        int f = ip[len] & 3;
        len++;
        static const uint32_t K[4] = {0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xca62c1d6};
        uint32_t A = ln(d, 3), B = ln(d, 2), C = ln(d, 1), D = ln(d, 0), E = 0;
        for (int i = 0; i < 4; i++) {
            uint32_t fn = f == 0 ? (B & C) ^ (~B & D) : f == 2 ? (B & C) ^ (B & D) ^ (C & D) : B ^ C ^ D;
            uint32_t a = fn + rol(A, 5) + ln(s, 3 - i) + E + K[f];
            E = D; D = C; C = rol(B, 30); B = A; A = a;
        }
        r[3] = A; r[2] = B; r[1] = C; r[0] = D;
    } else {
        signal(SIGILL, SIG_DFL);
        return;
    }
    for (int i = 0; i < 4; i++) putu32(d, i, r[i]);
    uc->uc_mcontext->__ss.__rip += (uint64_t)len;
}
static int emulate_sha(void) {
    struct sigaction sa;
    memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = sha_emulate;
    sa.sa_flags = SA_SIGINFO;
    return sigaction(SIGILL, &sa, 0) == 0;
}
#else
static int emulate_sha(void) { return 0; }
#endif
`

const vecHarnessTail = `};
static uint64_t seed = 0x9e3779b97f4a7c15ull;
static uint64_t next(void) { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; }
int main(void) {
    static const int counts[] = {0, 1, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200, 1000, -1};
    int have = features();
    boxes();
    if (!(have & HAS_SHA) && emulate_sha()) {
        printf("NOTE SHA-NI emulated: the processor lacks it, and SIGILL runs Intel's pseudocode\n");
        have |= HAS_SHA;
    }
    for (unsigned k = 0; k < sizeof cases / sizeof cases[0]; k++) {
        if (cases[k].needs & ~have) {
            printf("SKIP %s: the processor lacks its instructions\n", cases[k].name);
            continue;
        }
        for (int it = 0; it < 64; it++) {
            uint8_t a[16], b[16], c[16], got[16], want[16];
            for (int i = 0; i < 16; i += 8) { uint64_t x = next(), y = next(), z = next(); memcpy(a + i, &x, 8); memcpy(b + i, &y, 8); memcpy(c + i, &z, 8); }
            if (it < 8) { /* edges: lanes at their limits, and indices that probe swizzle's range */
                for (int i = 0; i < 16; i++) { a[i] = it & 1 ? 0x80 : 0x7f; b[i] = it & 2 ? 0xff : (uint8_t)(i * 3 + it); }
            }
            int n = counts[it % (sizeof counts / sizeof counts[0])];
            if (it >= 32) n = (int)next();
            memset(got, 0, 16); memset(want, 0, 16);
            cases[k].got(got, a, b, c, n);
            cases[k].want(want, a, b, c, n);
            if (memcmp(got, want, (size_t)cases[k].words) != 0) {
                printf("%s (n=%d):", cases[k].name, n);
                for (int i = 0; i < cases[k].words; i++) printf(" %02x/%02x", got[i], want[i]);
                printf("\n");
                break;
            }
        }
    }
    return 0;
}
`
"""

needs = {None: "0", "ir.CapAES": "HAS_AES", "ir.CapClmul": "HAS_CLMUL", "ir.CapSHA1": "HAS_SHA",
         "ir.CapSHA256": "HAS_SHA", "ir.CapCRC32C": "HAS_SSE42"}

go = ['''// Code generated by testdata/gen_vec_run_test.py; DO NOT EDIT.

package amd64_test

// §V, §V5 and §A8 lowered onto SSE2 and the extensions past it, and run:
// each verb against a scalar C reference of its definition. The
// cryptographic functions require their capability, which is how a library
// compiles its hardware path for a baseline target; the rest run at the
// baseline, and the verbs with an SSSE3 path run again in a function that
// requires aes. SHA-512, SM3, SM4 and the IEEE CRC-32 are not here: x86-64
// has no instruction for them, and LegalizeCryptoOn makes them calls. A
// case the processor running the test lacks the instructions for is logged
// and skipped, except SHA under Rosetta, which the harness emulates. TestRunVectorVerbs is
// generated by testdata/gen_vec_run_test.py; edit the list there.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunVectorVerbs(t *testing.T) {
	m := ir.NewModule("t", ir.X86_64MacOS)
	type verb struct {
		name string
		kind byte
		caps []ir.Capability
		body func(e *ir.Block, v ir.V128NS, a, b, c ir.V128, n ir.I32, x64 ir.I64, pa ir.Ptr) ir.Value
	}
	verbs := []verb{''']
for name, gobody, cref, kind, caps in T:
    capl = "nil" if not caps else "[]ir.Capability{%s}" % caps
    go.append('\t\t{"%s", \'%s\', %s, func(e *ir.Block, v ir.V128NS, a, b, c ir.V128, n ir.I32, x64 ir.I64, pa ir.Ptr) ir.Value {\n\t\t\treturn %s\n\t\t}},' % (name, kind, capl, gobody))
go.append('''	}
	for i, vb := range verbs {
		f := m.Func("_vt" + itoa(i)).Export().Requires(vb.caps...)
		out := f.ParamPtr("out")
		pa := f.ParamPtr("a")
		pb := f.ParamPtr("b")
		pc := f.ParamPtr("c")
		n := f.ParamI32("n")
		e := f.Entry()
		v := e.V128()
		a := v.Load(pa, ir.Align(1))
		b := v.Load(pb, ir.Align(1))
		c := v.Load(pc, ir.Align(1))
		x64 := e.I64.Load(pa, ir.Align(1))
		switch r := vb.body(e, v, a, b, c, n, x64, pa).(type) {
		case ir.V128:
			v.Store(r, out, ir.Align(1))
		case ir.I32:
			e.I32.Store(r, out, ir.Align(1))
		case ir.I64:
			e.I64.Store(r, out, ir.Align(1))
		}
		e.Return()
	}
	if err := m.Err(); err != nil {
		t.Fatal(err)
	}
	var c strings.Builder
	c.WriteString(vecHarnessHead)
	for i, vb := range verbs {
		c.WriteString("void vt" + itoa(i) + "(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int);\\n")
		c.WriteString("static void ref" + itoa(i) + "(uint8_t* r, const uint8_t* a, const uint8_t* b, const uint8_t* c, int n) { (void)a; (void)b; (void)c; (void)n; " + vecRefs[vb.name].ref + " }\\n")
	}
	c.WriteString("static const struct { const char* name; int words; int needs; void (*got)(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int); void (*want)(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int); } cases[] = {\\n")
	for i, vb := range verbs {
		words := "16"
		switch vb.kind {
		case 'w':
			words = "4"
		case 'x':
			words = "8"
		}
		c.WriteString("    {\\"" + vb.name + "\\", " + words + ", " + vecRefs[vb.name].needs + ", vt" + itoa(i) + ", ref" + itoa(i) + "},\\n")
	}
	c.WriteString(vecHarnessTail)
	var bad []string
	for _, line := range strings.Split(strings.TrimSpace(runNative(t, m, c.String())), "\\n") {
		switch {
		case line == "":
		case strings.HasPrefix(line, "SKIP "), strings.HasPrefix(line, "NOTE "):
			t.Log(line)
		default:
			bad = append(bad, line)
		}
	}
	if len(bad) > 0 {
		t.Errorf("verbs that disagree with their definition:\\n%s", strings.Join(bad, "\\n"))
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b []byte
	for i > 0 {
		b = append([]byte{byte('0' + i%10)}, b...)
		i /= 10
	}
	return string(b)
}

var vecRefs = map[string]struct{ ref, needs string }{''')
for name, gobody, cref, kind, caps in T:
    go.append('\t"%s": {`%s`, "%s"},' % (name, cref, needs[caps]))
go.append('''}
''')
go.append(HARNESS)
open(OUT, "w").write("\n".join(go))
print(len(T), "verbs")
