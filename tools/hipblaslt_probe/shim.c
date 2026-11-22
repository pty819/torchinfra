#include <hip/hip_runtime.h>
#include "hipblaslt/hipblaslt.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CK(x) do{ auto s=(x); if(s!=0){ printf("FAIL %s -> %d\n", #x, (int)s); exit(1);} }while(0)

int main(int argc, char** argv){
    int M = argc>1?atoi(argv[1]):7102, N = argc>2?atoi(argv[2]):12288, K = argc>3?atoi(argv[3]):4096;
    hipblasLtHandle_t h; CK(hipblasLtCreate(&h));
    void *A,*B,*D,*SA,*SB,*WS;
    hipMalloc(&A, (size_t)M*K); hipMalloc(&B, (size_t)N*K);
    hipMalloc(&D, (size_t)M*N*2); hipMalloc(&SA, M*4); hipMalloc(&SB, N*4);
    size_t wsz = 128<<20; hipMalloc(&WS, wsz);
    { char* ha = (char*)malloc(M*K); for(long i=0;i<(long)M*K;i++) ha[i] = (char)(rand()&0xff);
      char* hb = (char*)malloc(N*K); for(long i=0;i<(long)N*K;i++) hb[i] = (char)(rand()&0xff);
      hipMemcpy(A, ha, M*K, hipMemcpyHostToDevice); hipMemcpy(B, hb, N*K, hipMemcpyHostToDevice); free(ha); free(hb); }
    hipMemset(SA, 0, M*4); hipMemset(SB, 0, N*4);
    float one=1.0f; hipMemcpy(SA,&one,4,hipMemcpyHostToDevice); hipMemcpy(SB,&one,4,hipMemcpyHostToDevice);

    hipblasLtMatmulDesc_t op; CK(hipblasLtMatmulDescCreate(&op, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    hipblasOperation_t tA=HIPBLAS_OP_T, tB=HIPBLAS_OP_N;
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSA, &tA, 4));
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_TRANSB, &tB, 4));
    hipblasLtMatmulMatrixScale_t sm = HIPBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F;
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, 4));
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, 4));
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &SB, 8));
    CK(hipblasLtMatmulDescSetAttribute(op, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &SA, 8));
    hipblasLtMatrixLayout_t la,lb,lc,ldsc;
    CK(hipblasLtMatrixLayoutCreate(&la, HIP_R_8F_E4M3, K, N, K));
    CK(hipblasLtMatrixLayoutCreate(&lb, HIP_R_8F_E4M3, K, M, K));
    CK(hipblasLtMatrixLayoutCreate(&lc,  HIP_R_16BF, N, M, N));
    CK(hipblasLtMatrixLayoutCreate(&ldsc,HIP_R_16BF, N, M, N));
    hipblasLtMatmulPreference_t pref; CK(hipblasLtMatmulPreferenceCreate(&pref));
    uint64_t pws = wsz; CK(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &pws, 8));
    hipblasLtMatmulHeuristicResult_t heur[32]; int nret=0;
    CK(hipblasLtMatmulAlgoGetHeuristic(h, op, la, lb, lc, ldsc, pref, 32, heur, &nret));
    printf("heuristic: %d algos (M=%d K=%d N=%d)\n", nret, M, K, N);
    float alpha=1.0f, beta=0.0f;
    hipStream_t st = 0;
    double best_t = 1e9; int best_i = -1;
    for(int i=0;i<nret;i++){
        if(heur[i].state != HIPBLAS_STATUS_SUCCESS) continue;
        if(heur[i].workspaceSize > wsz) continue;
        hipblasStatus_t s = hipblasLtMatmul(h, op, &alpha, B, la, A, lb, &beta, D, lc, D, ldsc,
                                            &heur[i].algo, WS, wsz, st);
        if(s != 0){ printf("algo[%d] matmul st=%d\n", i, (int)s); continue; }
        if(hipDeviceSynchronize()!=hipSuccess){ printf("algo[%d] 同步失败(跳过)\n", i); continue; }
        for(int w=0;w<3;w++) hipblasLtMatmul(h, op, &alpha, B, la, A, lb, &beta, D, lc, D, ldsc, &heur[i].algo, WS, wsz, st);
        hipDeviceSynchronize();
        hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        for(int w=0;w<20;w++) hipblasLtMatmul(h, op, &alpha, B, la, A, lb, &beta, D, lc, D, ldsc, &heur[i].algo, WS, wsz, st);
        hipEventRecord(e1); hipEventSynchronize(e1);
        float ms; hipEventElapsedTime(&ms, e0, e1); ms/=20;
        double tf = 2.0*M*N*K/(ms*1e-3)/1e12;
        printf("algo[%2d]: %7.3fms  %6.1f TF  ws=%zu waves=%.2f\n", i, ms, tf, heur[i].workspaceSize, heur[i].wavesCount);
        if(ms < best_t){ best_t = ms; best_i = i; }
    }
    if(best_i>=0) printf("BEST: algo[%d] %.1f TF\n", best_i, 2.0*M*N*K/(best_t*1e-3)/1e12);
    return 0;
}
