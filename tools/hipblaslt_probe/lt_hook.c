#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
typedef void* VP; typedef int ST;
static ST (*real_desc_create)(VP*, int, int);
static ST (*real_desc_setattr)(VP, int, const void*, size_t);
static ST (*real_layout_create)(VP*, int, int32_t, int32_t, int64_t);
static ST (*real_heuristic)(VP, VP, VP, VP, VP, VP, VP, int, void*, int*);
static ST (*real_matmul)(VP, VP, const void*, const void*, VP, const void*, VP, const void*,
                         const void*, VP, void*, VP, const void*, void*, size_t, void*);
static void init(void){
    if(!real_desc_create){
        real_desc_create = dlsym(RTLD_NEXT, "hipblasLtMatmulDescCreate");
        real_desc_setattr = dlsym(RTLD_NEXT, "hipblasLtMatmulDescSetAttribute");
        real_layout_create = dlsym(RTLD_NEXT, "hipblasLtMatrixLayoutCreate");
        real_heuristic = dlsym(RTLD_NEXT, "hipblasLtMatmulAlgoGetHeuristic");
        real_matmul = dlsym(RTLD_NEXT, "hipblasLtMatmul");
    }
}
ST hipblasLtMatmulDescCreate(VP* d, int c, int s){
    init(); ST r = real_desc_create(d, c, s);
    fprintf(stderr, "[hook] DescCreate compute=%d scale=%d -> %p (st=%d)\n", c, s, *d, r);
    return r;
}
ST hipblasLtMatmulDescSetAttribute(VP d, int attr, const void* v, size_t sz){
    init();
    if(sz == 4) fprintf(stderr, "[hook] DescSet %p attr=%d i32=%d\n", d, attr, *(const int*)v);
    else if(sz == 8) fprintf(stderr, "[hook] DescSet %p attr=%d ptr=%p\n", d, attr, *(void* const*)v);
    return real_desc_setattr(d, attr, v, sz);
}
ST hipblasLtMatrixLayoutCreate(VP* l, int dt, int32_t r, int32_t c, int64_t ld){
    init(); ST st = real_layout_create(l, dt, r, c, ld);
    fprintf(stderr, "[hook] Layout %p dt=%d rows=%d cols=%d ld=%ld (st=%d)\n", *l, dt, r, c, (long)ld, st);
    return st;
}
ST hipblasLtMatmulAlgoGetHeuristic(VP h, VP op, VP la, VP lb, VP lc, VP ldsc, VP pref, int req, void* out, int* ret){
    init(); ST st = real_heuristic(h, op, la, lb, lc, ldsc, pref, req, out, ret);
    fprintf(stderr, "[hook] Heuristic req=%d ret=%d st=%d\n", req, ret ? *ret : -1, st);
    if(ret && *ret > 0){
        uint8_t* a = (uint8_t*)out;
        fprintf(stderr, "[hook]   algo0 bytes:");
        for(int i=0;i<16;i++) fprintf(stderr, " %02x", a[i]);
        fprintf(stderr, "\n");
    }
    return st;
}
ST hipblasLtMatmul(VP h, VP op, const void* al, const void* A, VP la, const void* B, VP lb,
                   const void* be, const void* C, VP lc, void* D, VP ldsc, const void* algo,
                   void* ws, size_t wsz, void* stream){
    init();
    const uint8_t* a = (const uint8_t*)algo;
    fprintf(stderr, "[hook] Matmul A=%p B=%p C=%p D=%p algo_ptr=%p bytes:", A, B, C, D, algo);
    if(a) for(int i=0;i<16;i++) fprintf(stderr, " %02x", a[i]);
    fprintf(stderr, " ws=%zu\n", wsz);
    { FILE* f = fopen("/tmp/lt_cap.bin", "w");
      if(f){ fwrite(&h, 8, 1, f); fwrite(algo, 24, 1, f); fwrite(&wsz, 8, 1, f); fclose(f); } }
    return real_matmul(h, op, al, A, la, B, lb, be, C, lc, D, ldsc, algo, ws, wsz, stream);
}
