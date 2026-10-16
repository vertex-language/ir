# Generates ir/lower/arm64/vec_run_test.go: every §V verb against a scalar C
# reference of its definition, and every §V5 verb the M1 has against its own
# instruction, built natively and run.
import sys
OUT = sys.argv[1]
T = []  # (name, go body returning (kind, expr), C reference, requires caps)
# go body uses: e (*ir.Block), v (V128NS), a, b, c (ir.V128), n (ir.I32), x64 (ir.I64)
# result kinds: "v" -> V128 stored to out, "w" -> I32 stored, "x" -> I64 stored
def add(name, go, cref, kind="v", caps=None):
    T.append((name, go, cref, kind, caps))

def lanes(t, fn):  # C loop over lanes producing r from a,b
    return fn

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
# §V5 against the instructions themselves
Z = "vdupq_n_u8(0)"
add("aes_enc", "v.AesEnc(a, b)", "vst1q_u8(r, veorq_u8(vaesmcq_u8(vaeseq_u8(vld1q_u8(a), %s)), vld1q_u8(b)));"%Z, caps="ir.CapAES")
add("aes_enc_last", "v.AesEncLast(a, b)", "vst1q_u8(r, veorq_u8(vaeseq_u8(vld1q_u8(a), %s), vld1q_u8(b)));"%Z, caps="ir.CapAES")
add("aes_dec", "v.AesDec(a, b)", "vst1q_u8(r, veorq_u8(vaesimcq_u8(vaesdq_u8(vld1q_u8(a), %s)), vld1q_u8(b)));"%Z, caps="ir.CapAES")
add("aes_dec_last", "v.AesDecLast(a, b)", "vst1q_u8(r, veorq_u8(vaesdq_u8(vld1q_u8(a), %s), vld1q_u8(b)));"%Z, caps="ir.CapAES")
add("aes_inv_mix_columns", "v.AesInvMixColumns(a)", "vst1q_u8(r, vaesimcq_u8(vld1q_u8(a)));", caps="ir.CapAES")
add("aes_keygen_assist", "v.AesKeygenAssist(a, 0x1b)", "keygen(r, a, 0x1b);", caps="ir.CapAES")
for sel in range(4):
    add("i64x2_clmul_%d"%sel, "v.I64x2Clmul(a, b, %d)"%sel, "clmul(r, a, b, %d);"%sel, caps="ir.CapClmul")
add("sha1_rounds_ch", "v.Sha1RoundsCh(a, n, b)", "vst1q_u32((uint32_t*)r, vsha1cq_u32(vld1q_u32((const uint32_t*)a), (uint32_t)n, vld1q_u32((const uint32_t*)b)));", caps="ir.CapSHA1")
add("sha1_rounds_parity", "v.Sha1RoundsParity(a, n, b)", "vst1q_u32((uint32_t*)r, vsha1pq_u32(vld1q_u32((const uint32_t*)a), (uint32_t)n, vld1q_u32((const uint32_t*)b)));", caps="ir.CapSHA1")
add("sha1_rounds_maj", "v.Sha1RoundsMaj(a, n, b)", "vst1q_u32((uint32_t*)r, vsha1mq_u32(vld1q_u32((const uint32_t*)a), (uint32_t)n, vld1q_u32((const uint32_t*)b)));", caps="ir.CapSHA1")
add("sha1_msg1", "v.Sha1Msg1(a, b, c)", "vst1q_u32((uint32_t*)r, vsha1su0q_u32(vld1q_u32((const uint32_t*)a), vld1q_u32((const uint32_t*)b), vld1q_u32((const uint32_t*)c)));", caps="ir.CapSHA1")
add("sha1_msg2", "v.Sha1Msg2(a, b)", "vst1q_u32((uint32_t*)r, vsha1su1q_u32(vld1q_u32((const uint32_t*)a), vld1q_u32((const uint32_t*)b)));", caps="ir.CapSHA1")
add("sha256_rounds_abcd", "v.Sha256RoundsAbcd(a, b, c)", "vst1q_u32((uint32_t*)r, vsha256hq_u32(vld1q_u32((const uint32_t*)a), vld1q_u32((const uint32_t*)b), vld1q_u32((const uint32_t*)c)));", caps="ir.CapSHA256")
add("sha256_rounds_efgh", "v.Sha256RoundsEfgh(a, b, c)", "vst1q_u32((uint32_t*)r, vsha256h2q_u32(vld1q_u32((const uint32_t*)a), vld1q_u32((const uint32_t*)b), vld1q_u32((const uint32_t*)c)));", caps="ir.CapSHA256")
add("sha256_msg1", "v.Sha256Msg1(a, b)", "vst1q_u32((uint32_t*)r, vsha256su0q_u32(vld1q_u32((const uint32_t*)a), vld1q_u32((const uint32_t*)b)));", caps="ir.CapSHA256")
add("sha256_msg2", "v.Sha256Msg2(a, b, c)", "vst1q_u32((uint32_t*)r, vsha256su1q_u32(vld1q_u32((const uint32_t*)a), vld1q_u32((const uint32_t*)b), vld1q_u32((const uint32_t*)c)));", caps="ir.CapSHA256")
add("sha512_rounds_1", "v.Sha512Rounds1(a, b, c)", "vst1q_u64((uint64_t*)r, vsha512hq_u64(vld1q_u64((const uint64_t*)a), vld1q_u64((const uint64_t*)b), vld1q_u64((const uint64_t*)c)));", caps="ir.CapSHA512")
add("sha512_rounds_2", "v.Sha512Rounds2(a, b, c)", "vst1q_u64((uint64_t*)r, vsha512h2q_u64(vld1q_u64((const uint64_t*)a), vld1q_u64((const uint64_t*)b), vld1q_u64((const uint64_t*)c)));", caps="ir.CapSHA512")
add("sha512_msg1", "v.Sha512Msg1(a, b)", "vst1q_u64((uint64_t*)r, vsha512su0q_u64(vld1q_u64((const uint64_t*)a), vld1q_u64((const uint64_t*)b)));", caps="ir.CapSHA512")
add("sha512_msg2", "v.Sha512Msg2(a, b, c)", "vst1q_u64((uint64_t*)r, vsha512su1q_u64(vld1q_u64((const uint64_t*)a), vld1q_u64((const uint64_t*)b), vld1q_u64((const uint64_t*)c)));", caps="ir.CapSHA512")
for verb, ref, C in [("Crc32cU8","__crc32cb((uint32_t)n, a[0])","ir.CapCRC32C"),("Crc32cU16","__crc32ch((uint32_t)n, getu16(a,0))","ir.CapCRC32C"),
                      ("Crc32cU32","__crc32cw((uint32_t)n, getu32(a,0))","ir.CapCRC32C"),("Crc32U8","__crc32b((uint32_t)n, a[0])","ir.CapCRC32"),
                      ("Crc32U16","__crc32h((uint32_t)n, getu16(a,0))","ir.CapCRC32"),("Crc32U32","__crc32w((uint32_t)n, getu32(a,0))","ir.CapCRC32")]:
    add(verb.lower(), "e.I32.%s(n, e.I32.Load(pa))" % verb, "putu32(r,0,%s);" % ref, kind="w", caps=C)
add("crc32cu64", "e.I32.Crc32cU64(n, x64)", "putu32(r,0,__crc32cd((uint32_t)n, getu64(a,0)));", kind="w", caps="ir.CapCRC32C")
add("crc32u64", "e.I32.Crc32U64(n, x64)", "putu32(r,0,__crc32d((uint32_t)n, getu64(a,0)));", kind="w", caps="ir.CapCRC32")

go = ['''package arm64_test

// §V and §V5 lowered onto Advanced SIMD and run: each verb against a scalar
// C reference of its definition (SSE2's, which §V's are), and each
// cryptographic verb against the M1's own instruction through arm_neon.h.
// The cryptographic functions require their capability, which is how a
// library compiles its hardware path for a baseline target. SM3 and SM4
// are not here: the M1 has neither, and table_simd's clang check is what
// covers their encoding. Generated; see the scratch generator's history in
// the commit that added it.

import (
	"strings"
	"testing"

	"github.com/vertex-language/ir"
)

func TestRunVectorVerbs(t *testing.T) {
	m := ir.NewModule("t", ir.AArch64Linux)
	type verb struct {
		name string
		kind byte
		caps []ir.Capability
		body func(e *ir.Block, v ir.V128NS, a, b, c ir.V128, n ir.I32, x64 ir.I64, pa ir.Ptr) ir.Value
	}
	verbs := []verb{''']
for name, gobody, cref, kind, caps in T:
    capl = "nil" if not caps else "[]ir.Capability{%s}" % caps
    go.append('\t\t{"%s", \'%s\', %s, func(e *ir.Block, v ir.V128NS, a, b, c ir.V128, n ir.I32, x64 ir.I64, pa ir.Ptr) ir.Value { return %s }},' % (name, kind, capl, gobody))
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
		c.WriteString("static void ref" + itoa(i) + "(uint8_t* r, const uint8_t* a, const uint8_t* b, const uint8_t* c, int n) { (void)a; (void)b; (void)c; (void)n; " + vecRefs[vb.name] + " }\\n")
	}
	c.WriteString("static const struct { const char* name; int words; void (*got)(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int); void (*want)(uint8_t*, const uint8_t*, const uint8_t*, const uint8_t*, int); } cases[] = {\\n")
	for i, vb := range verbs {
		words := "16"
		switch vb.kind {
		case 'w':
			words = "4"
		case 'x':
			words = "8"
		}
		c.WriteString("    {\\"" + vb.name + "\\", " + words + ", vt" + itoa(i) + ", ref" + itoa(i) + "},\\n")
	}
	c.WriteString(vecHarnessTail)
	if got := runNative(t, m, c.String()); got != "" {
		t.Errorf("verbs that disagree with their definition:\\n%s", got)
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

var vecRefs = map[string]string{''')
for name, gobody, cref, kind, caps in T:
    go.append('\t"%s": %s,' % (name, '`' + cref + '`'))
go.append('''}

const vecHarnessHead = `#include <arm_neon.h>
#include <arm_acle.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
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
static void clmul(uint8_t* r, const uint8_t* a, const uint8_t* b, int sel) {
    poly128_t p = vmull_p64(getu64(a, sel & 1), getu64(b, sel >> 1)); memcpy(r, &p, 16);
}
static uint32_t sb(uint32_t w) {
    uint8x16_t x = vreinterpretq_u8_u32(vdupq_n_u32(w));
    /* SubBytes of w's four bytes: AESE against zero is SR(SB(x)), and a
       column of four equal words is fixed by ShiftRows. */
    return vgetq_lane_u32(vreinterpretq_u32_u8(vaeseq_u8(x, vdupq_n_u8(0))), 0);
}
static void keygen(uint8_t* r, const uint8_t* a, int rcon) {
    uint32_t x1 = sb(getu32(a, 1)), x3 = sb(getu32(a, 3));
    putu32(r, 0, x1); putu32(r, 1, ((x1 >> 8) | (x1 << 24)) ^ (uint32_t)rcon);
    putu32(r, 2, x3); putu32(r, 3, ((x3 >> 8) | (x3 << 24)) ^ (uint32_t)rcon);
}
`

const vecHarnessTail = `};
static uint64_t seed = 0x9e3779b97f4a7c15ull;
static uint64_t next(void) { seed ^= seed << 13; seed ^= seed >> 7; seed ^= seed << 17; return seed; }
int main(void) {
    static const int counts[] = {0, 1, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 200, 1000, -1};
    for (unsigned k = 0; k < sizeof cases / sizeof cases[0]; k++) {
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
                printf("\\\\n");
                break;
            }
        }
    }
    return 0;
}
`
''')
open(OUT, "w").write("\n".join(go))
print(len(T), "verbs")
