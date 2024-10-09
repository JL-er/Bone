import torch
from einops import rearrange
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BM': 64}, num_stages=2, num_warps=2),
        triton.Config({'BM': 64}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128}, num_stages=2, num_warps=2),
        triton.Config({'BM': 128}, num_stages=2, num_warps=4),
        triton.Config({'BM': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 64}, num_stages=4, num_warps=2),
        triton.Config({'BM': 64}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128}, num_stages=4, num_warps=2),
        triton.Config({'BM': 128}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128}, num_stages=4, num_warps=8),
        triton.Config({'BM': 256}, num_stages=2, num_warps=2),
        triton.Config({'BM': 256}, num_stages=2, num_warps=4),
        triton.Config({'BM': 256}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256}, num_stages=4, num_warps=2),
        triton.Config({'BM': 256}, num_stages=4, num_warps=4),
        triton.Config({'BM': 256}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)

@triton.jit
def bone_fwd_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    bone,
    # Matrix dimensions
    M,
    N,
    K,
    s_am,
    s_ak,
    s_bk,
    s_bn,
    s_cm,
    s_cn,
    s_bonep,
    s_bonem,
    s_bonen,
    # Meta-parameters
    #BP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    NM, NN = tl.num_programs(0), tl.num_programs(1)
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    i_m, i_n = tl.swizzle2d(i_m, i_n,  NM, NN, G)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details

    o_am = (i_m * BM + tl.arange(0, BM))
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[ :, None] * s_bk + o_bn[None, :] * s_bn)

    p_bone = bone + i_n * s_bonep + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen
    b_bone = tl.load(p_bone)


    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_b += tl.dot(b_b, b_bone, allow_tf32=False).to(b_b.dtype)+b_bone.to(b_b.dtype)
        #b_acc = b_b
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    mask = (o_cn[None, :] < N )

    #b_c = b_acc

    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_acc.to(c.dtype.element_ty), mask=mask)





#@contiguous
def bone_fwd(
    a: torch.Tensor,
    b: torch.Tensor,
    bone: torch.Tensor,
) -> torch.Tensor:
    #assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    B, L, K = a.shape
    M = B*L
    K, N = b.shape
    # Allocates output.
    c = a.new_empty(B, L, N)
    # print(c.shape,c.dtype)
    # print(N//64)
    # BM=64
    BK=BN = 64

    #grid=(triton.cdiv(M, BM), triton.cdiv(N, BN))
    def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, BN))
    bone_fwd_kernel[grid](
        a, b, c, bone, 
        M, N, K,
        a.stride(1), a.stride(2),
        b.stride(0), b.stride(1),
        c.stride(1), c.stride(2),
        bone.stride(0), bone.stride(1), bone.stride(2),
        BK=BK,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c



@triton.autotune(
    configs=[
        triton.Config({'BM': 64}, num_stages=2, num_warps=2),
        triton.Config({'BM': 64}, num_stages=2, num_warps=4),
        triton.Config({'BM': 64}, num_stages=2, num_warps=8),
        triton.Config({'BM': 128}, num_stages=2, num_warps=2),
        triton.Config({'BM': 128}, num_stages=2, num_warps=4),
        triton.Config({'BM': 128}, num_stages=2, num_warps=8),
        triton.Config({'BM': 64}, num_stages=4, num_warps=2),
        triton.Config({'BM': 64}, num_stages=4, num_warps=4),
        triton.Config({'BM': 64}, num_stages=4, num_warps=8),
        triton.Config({'BM': 128}, num_stages=4, num_warps=2),
        triton.Config({'BM': 128}, num_stages=4, num_warps=4),
        triton.Config({'BM': 128}, num_stages=4, num_warps=8),
        triton.Config({'BM': 256}, num_stages=2, num_warps=2),
        triton.Config({'BM': 256}, num_stages=2, num_warps=4),
        triton.Config({'BM': 256}, num_stages=2, num_warps=8),
        triton.Config({'BM': 256}, num_stages=4, num_warps=2),
        triton.Config({'BM': 256}, num_stages=4, num_warps=4),
        triton.Config({'BM': 256}, num_stages=4, num_warps=8),
        triton.Config({'BM': 256}, num_stages=1, num_warps=2),
        triton.Config({'BM': 256}, num_stages=1, num_warps=4),
        triton.Config({'BM': 256}, num_stages=1, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bone_gradx_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    bone,
    # Matrix dimensions
    M,
    N,
    K,
    s_am,
    s_ak,
    s_bk,
    s_bn,
    s_cm,
    s_cn,
    s_bonep,
    s_bonem,
    s_bonen,
    # Meta-parameters
    #BP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    NM, NN = tl.num_programs(0), tl.num_programs(1)
    i_m, i_n = tl.program_id(0), tl.program_id(1)
    i_m, i_n = tl.swizzle2d(i_m, i_n,  NM, NN, G)

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `p_a` is a block of [BM, BK] pointers
    # `p_b` is a block of [BK, BN] pointers
    # See above `Pointer Arithmetic` section for details

    o_am = (i_m * BM + tl.arange(0, BM))
    o_bn = (i_n * BN + tl.arange(0, BN)) % N
    o_k = tl.arange(0, BK)

    p_a = a + (o_am[:, None] * s_am + o_k[None, :] * s_ak)
    p_b = b + (o_k[ :, None] * s_bk + o_bn[None, :] * s_bn)

    p_bone = bone + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen


    b_acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        b_bone = tl.load(p_bone)
        b_a = tl.load(p_a, mask=o_k[None, :] < K - k * BK, other=0.0)
        b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)
        b_bone = tl.dot(b_bone,b_b, allow_tf32=False).to(b_b.dtype)+b_bone


        b_b = b_b+b_bone
        #b_acc = b_b
        # We accumulate along the K dimension.
        b_acc += tl.dot(b_a, b_b, allow_tf32=False)
        # Advance the ptrs to the next K block.
        p_a += BK * s_ak
        p_b += BK * s_bk
        p_bone += s_bonep

    o_cm = i_m * BM + tl.arange(0, BM)
    o_cn = i_n * BN + tl.arange(0, BN)
    #mask =  ( o_cm[:, None] < M) & (o_cn[None, :] < N )

    #b_c = b_acc

    p_c = c + s_cm * o_cm[:, None] + s_cn * o_cn[None, :]

    tl.store(p_c, b_acc.to(c.dtype.element_ty) )




def bone_gradx(
    do: torch.Tensor,
    b: torch.Tensor,
    bone: torch.Tensor,

) -> torch.Tensor:
    #assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    B, L, K = do.shape
    M = B*L
    K, N = b.shape
    _,block,_ =bone.shape
    # Allocates output.
    c = do.new_empty(B, L, N)
    BK=BN = block


    def grid(meta): return (triton.cdiv(M, meta['BM']), triton.cdiv(N, BN))
    bone_gradx_kernel[grid](
        do, b, c, bone, 
        M, N, K,
        do.stride(1), do.stride(2),
        b.stride(0), b.stride(1),
        c.stride(1), c.stride(2),
        bone.stride(0), bone.stride(1), bone.stride(2),
        BK=BK,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c


@triton.autotune(
    configs=[
        triton.Config({'BK': 32}, num_stages=2, num_warps=2),
        triton.Config({'BK': 32}, num_stages=2, num_warps=4),
        triton.Config({'BK': 32}, num_stages=2, num_warps=8),
        triton.Config({'BK': 32}, num_stages=4, num_warps=2),
        triton.Config({'BK': 32}, num_stages=4, num_warps=4),
        triton.Config({'BK': 32}, num_stages=4, num_warps=8),
        triton.Config({'BK': 64}, num_stages=1, num_warps=2),
        triton.Config({'BK': 64}, num_stages=1, num_warps=4),
        triton.Config({'BK': 64}, num_stages=1, num_warps=8),
        triton.Config({'BK': 64}, num_stages=2, num_warps=2),
        triton.Config({'BK': 64}, num_stages=2, num_warps=4),
        triton.Config({'BK': 64}, num_stages=2, num_warps=8),
        triton.Config({'BK': 128}, num_stages=2, num_warps=2),
        triton.Config({'BK': 128}, num_stages=2, num_warps=4),
        triton.Config({'BK': 128}, num_stages=2, num_warps=8),
        # triton.Config({'BK': 64}, num_stages=4, num_warps=2),
        # triton.Config({'BK': 64}, num_stages=4, num_warps=4),
        # triton.Config({'BK': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BK': 128}, num_stages=4, num_warps=2),
        # triton.Config({'BK': 128}, num_stages=4, num_warps=4),
        # triton.Config({'BK': 128}, num_stages=4, num_warps=8),
        # triton.Config({'BK': 256}, num_stages=2, num_warps=2),
        # triton.Config({'BK': 256}, num_stages=2, num_warps=4),
        # triton.Config({'BK': 256}, num_stages=2, num_warps=8),
        # triton.Config({'BK': 256}, num_stages=4, num_warps=2),
        # triton.Config({'BK': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BK': 256}, num_stages=4, num_warps=8),
    ],
    key=['M','N','K'],
)

@triton.jit
def bone_gradb_kernel(
    # Pointers to matrices
    a,
    b,
    c,
    w,
    # Matrix dimensions
    BL,
    M,
    N,
    K,
    s_ab,
    s_am,
    s_ak,
    s_bb,
    s_bk,
    s_bn,
    s_cp,
    s_cm,
    s_cn,
    s_wm,
    s_wn,

    # Meta-parameters
    #BP: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    G: tl.constexpr,
    ACTIVATION: tl.constexpr,
):

    i_n = tl.program_id(0)

    offs_B = i_n//BL
    o_bn = (i_n * BN + tl.arange(0, BN))%N
    o_k = tl.arange(0, BK)
    o_m = tl.arange(0, BM)
    o_block=tl.arange(0, 64)
    o_wn = o_bn%N

    p_a = a + (o_m[:, None] * s_am + o_k[None, :] * s_ak + offs_B*s_ab)
    p_b = b + (o_k[ :, None] * s_bk + o_bn[None, :] * s_bn + offs_B*s_bb)

    p_w = w + s_wm * o_block[:, None] + s_wn * o_wn[None, :]

    #p_bone = bone + o_k[ :, None] * s_bonem +  o_k[None, :] * s_bonen + i_n * s_bonep
    dc = tl.zeros((64, 64), dtype=tl.float32)
    for m in range(0, tl.cdiv(M, BM)):
        b_dw = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, tl.cdiv(K, BK)):

            b_a = tl.load(p_a, mask=(o_k[None, :] < K - k * BK), other=0.0)
            b_b = tl.load(p_b, mask=o_k[:, None] < K - k * BK, other=0.0)

            b_dw += tl.dot(b_a, b_b, allow_tf32=False)
            # Advance the ptrs to the next K block.
            p_a += BK * s_ak
            p_b += BK * s_bk

        b_w = tl.load(p_w)
        p_a += BM * s_am
        p_w += BM * s_wm
        p_a -= K * s_ak
        p_b -= K * s_bk
        dc += tl.dot(b_w.T, b_dw.to(b_w.dtype), allow_tf32=False).to(b_w.dtype)+b_dw
        #dc += b_dw


    p_c = c + o_block[ :, None] * s_cm +  o_block[None, :] * s_cn + i_n*s_cp

    tl.store(p_c, dc.to(c.dtype.element_ty) )




def bone_gradb(
    x: torch.Tensor,
    do: torch.Tensor,
    w: torch.Tensor,
    bone_g: int,
    bone_b: int,

) -> torch.Tensor:
    #assert a.shape[2] == b.shape[1], 'Incompatible dimensions (A: {}x{}x{}, B: {}x{}x{})'.format(*a.shape, *b.shape)

    B, M, K = x.shape
    _, K, O = do.shape
    N = B*O
    
    c = torch.zeros((B, bone_g, bone_b, bone_b), dtype=a.dtype, device=a.device)
    BM = BN = bone_b
    BL = triton.cdiv(O, BN)

    grid= (triton.cdiv(N, BN),)
    bone_gradb_kernel[grid](
        x, do, c, w, 
        BL, M, O, K,
        x.stride(0), x.stride(1), x.stride(2),
        do.stride(0),do.stride(1), do.stride(2),
        c.stride(1), c.stride(2), c.stride(3),
        w.stride(0), w.stride(1),
        BM=BM,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c

# def native_gradb(x, do, w):
#     w = rearrange(w, '(a r1) (b r2)->a b r1 r2 ', r1=64, r2=64)
#     dw = x.transpose(1,2)@do
#     dw = torch.sum(rearrange(dw, 'q (a r1) (b r2) ->q a b r1 r2 ', r1=64,r2=64), dim=0)
#     tc = torch.sum(w.transpose(-2,-1)@dw, dim=0)
#     tc += torch.sum(dw, dim=0)
#     return tc

def sum_gradb(x, do, w):

    dw = x.transpose(1,2)@do
    dw = rearrange(dw, 'q (a r1) (b r2) ->q a b r1 r2 ', r1=64,r2=64)
    ww = rearrange(w, '(a r1) (b r2) ->a b r1 r2 ', r1=64,r2=64)
    tc  = torch.sum(ww.transpose(-2,-1)@dw, dim=1)
    tc += torch.sum(dw, dim=1)
    tc =torch.sum(tc, dim=0)
    return tc

class BoneTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b):

        o = bone_fwd(x,w,b)
        ctx.save_for_backward(x, w, b)
        return o

    @staticmethod
    def backward(ctx, do):
        x, w, b= ctx.saved_tensors
        bone_g,bone_b,_=b.shape
        dx = bone_gradx(do, w.t(), b.transpose(1,2))

        dc = bone_gradb(x.transpose(1,2), do, w, bone_g, bone_b)
        dc = torch.sum(dc, dim=0)

        return dx, None, dc



def flash_bone(x,w,b):
    o = BoneTriton.apply(x,w,b)
    return o

def bone(x,w,b):
    ww = rearrange(w, '(a r1) (b r2) -> a b r1 r2', r1 = 64 , r2 = 64 )@b+b
    ww= rearrange(ww, 'a b r1 r2 ->(a r1) (b r2) ')+w
    return x@ww


import torch
from torch.autograd import gradcheck
#torch.manual_seed(49)
import time
dtype=torch.bfloat16
block_size = 64
B, L, input, ouput, bone_b = 8, 512, 2560,2560 , 64
a = torch.randn((B, L,input), dtype=dtype, requires_grad=True, device='cuda')

b = torch.randn((input,ouput), dtype=dtype, requires_grad=True, device='cuda')
c = torch.randn((ouput//bone_b, bone_b,bone_b), dtype=dtype, requires_grad=True, device='cuda')
do = torch.randn((B, L,ouput), dtype=dtype, requires_grad=True, device='cuda')
a1=a.clone()
b1=b.clone()
c1=c.clone()
do1 = do.clone()

o = flash_bone(a,b,c)
o.backward(do)
da, a.grad = a.grad.clone(), None
dc, c.grad = c.grad.clone(), None
torch.cuda.synchronize()

ts = 0
for i in range(100):
    o = flash_bone(a,b,c)
    s = time.time()
    o.backward(do)
    torch.cuda.synchronize()
    e = time.time()
    ts+=(e-s)
print('bone flash ', ts/100)


ts = 0

for i in range(100):
    o1 = bone(a,b,c)
    s = time.time()
    o1.backward(do)

    torch.cuda.synchronize()
    e = time.time()
    ts += (e-s)
print('bone navie ', ts/100)





da, a.grad = a.grad.clone(), None
dc, c.grad = c.grad.clone(), None


# w = rearrange(b, '(a r1) (b r2) -> a b r1 r2', r1 = 64 , r2 = 64 )@c+c
# ww= rearrange(w, 'a b r1 r2 ->(a r1) (b r2) ')+b

# stc = sum_gradb(a, do, b)
# #torch.testing.assert_close(tc, stc, rtol=1e-2, atol=1e-8)

# # bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')

# dcc = bone_gradb(a.transpose(1,2), do, b, ouput//bone_b ,bone_b)

# dcc = torch.sum(dcc, dim=0)
# print(stc.view(-1))
# print(dcc.view(-1))

# close = torch.allclose(tc, dcc.to(tc.dtype), rtol=1e-2, atol=1e-8, equal_nan=False)
# bb = rearrange(b, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
# dw = a.t()@do
# dw = rearrange(dw, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
# #dw = rearrange(dw, 'a r1 (b r2) -> a b r1 r2 ', r2=64)
# tc  = torch.sum(bb.transpose(2,3)@dw, dim=0)
# tc += torch.sum(dw, dim=0)
# #tc = dw
# print(tc.reshape(-1))
# #bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')

# dcc = bone_bwd_wb(a.t(), do, b, 40 ,64)

# close = torch.allclose(tc, dcc.to(tc.dtype), rtol=1e-03, atol=1e-6, equal_nan=False)
# print('grad sum',close)
# print(dcc.reshape(-1))

# torch.testing.assert_close(tc, dcc.to(tc.dtype), rtol=1e-3, atol=1e-8)

# bb = rearrange(b, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)
# dw = a.t()@do
# dw = rearrange(dw, '(a r1) (b r2) -> a b r1 r2 ', r1=64,r2=64)

# tc  = torch.sum(bb.transpose(2,3)@dw, dim=0)
# tc += torch.sum(dw, dim=0)
# print(tc.reshape(-1))

# dc = bone_bwd_wb(a.t(), do, b, 40 ,64)
# print(dc.reshape(-1))

# torch.testing.assert_close(tc, dc.to(tc.dtype), rtol=1e-5, atol=1e-8)



print('forward', torch.allclose(o, o1, rtol=1e-2, atol=1e-08, equal_nan=False))
close = torch.allclose(da, a.grad, rtol=1e-2, atol=1e-08, equal_nan=False)
print('grad a',close)

# close = torch.allclose(db, b.grad, rtol=1e-05, atol=1e-8, equal_nan=False)
# print('grad b',close)

close = torch.allclose(dc, c.grad, rtol=1e-05, atol=1e-4, equal_nan=False)
print('grad c',close)
print(c.grad.shape,c.grad.reshape(B, -1))
print(dc.shape,dc.reshape(B, -1))
#torch.testing.assert_close(o, o1, rtol=1e-4, atol=1e-8)

#print(torch.allclose(da, a.grad, rtol=1e-05, atol=1e-4, equal_nan=False))
# torch.testing.assert_close(da, a.grad, rtol=1e-4, atol=1e-8)
torch.testing.assert_close(dc, c.grad, rtol=1e-2, atol=1e-8)
