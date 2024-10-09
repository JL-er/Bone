import torch
from einops import rearrange
import triton
import triton.language as tl


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
def bone_gradwb(
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




def bone_bwd_wb(
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
    bone_gradwb[grid](
        x, do, c, w, 
        BL, M, O, K,
        x.stride(0), x.stride(1), x.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        c.stride(1), c.stride(2), c.stride(3),
        w.stride(0), w.stride(1),
        BM=BM,BN=BN,G=4,
        ACTIVATION=None,
    )
    return c

# 使用方法
def bone(a,b, c):
    o = CustomEinsum.apply(a, b,c)
    return o


import torch
from torch.autograd import gradcheck
import time
# 创建输入张量
dtype=torch.bfloat16
B = 32
a = torch.randn((B, 512,64*20), dtype=dtype, requires_grad=True, device='cuda')

b = torch.randn((20,40,64,64), dtype=dtype, requires_grad=True, device='cuda')
c = torch.randn((40,64,64), dtype=dtype, requires_grad=True, device='cuda')
e = torch.randn((128,128), dtype=dtype, requires_grad=True, device='cuda')
do = torch.randn((B, 512,2560), dtype=dtype, requires_grad=True, device='cuda')

# block_size = 64
# B, L, input, ouput, bone_b = 4, 2048, 2560,2560 , 64
# a = torch.randn((B, L,input), dtype=dtype, requires_grad=True, device='cuda')

# b = torch.randn((40,40,64,64), dtype=dtype, requires_grad=True, device='cuda')
# #c = torch.randn((ouput//bone_b, bone_b,bone_b), dtype=dtype, requires_grad=True, device='cuda')
# do = torch.randn((B, L,ouput), dtype=dtype, requires_grad=True, device='cuda')


def native_gradb(x, do, w):

    dw = x.transpose(1,2)@do
    dw = rearrange(dw, 'q (a r1) (b r2) ->q a b r1 r2 ', r1=64,r2=64)
    tc  = torch.sum(w.transpose(-2,-1)@dw, dim=1)
    tc += torch.sum(dw, dim=1)
    return tc
tc = native_gradb(a, do, b)

def sum_gradb(x, do, w):

    dw = torch.sum(x.transpose(1,2)@do, dim=0)
    dw = rearrange(dw, '(a r1) (b r2) ->a b r1 r2 ', r1=64,r2=64)
    tc  = torch.sum(w.transpose(-2,-1)@dw, dim=0)
    tc += torch.sum(dw, dim=0)
    return tc
tc =torch.sum(tc, dim=0)
stc = sum_gradb(a, do, b)
#torch.testing.assert_close(tc, stc, rtol=1e-2, atol=1e-8)

bb = rearrange(b, 'a b r1 r2 ->(a r1) (b r2) ')

dcc = bone_bwd_wb(a.transpose(1,2), do, bb, 40 ,64)
dcc = torch.sum(dcc, dim=0)
print(stc.view(-1))
print(dcc.view(-1))

close = torch.allclose(tc, dcc.to(tc.dtype), rtol=1e-2, atol=1e-8, equal_nan=False)
print('grad sum',close)
print(dcc.to(tc.dtype).reshape(B,-1))

#torch.testing.assert_close(stc, dcc.to(tc.dtype), rtol=1e-2, atol=1e-8)

s = time.time()
for i in range(100):
    tc = native_gradb(a, do, b)
torch.cuda.synchronize()
e = time.time()
print('bone navie ', (e-s)/100)

bbb = bb.t()
cc = c.transpose(1,2)
aa = a.transpose(1,2)
s = time.time()
for i in range(100):
    dcc = bone_bwd_wb(aa, do, bb, 40 ,64)
torch.cuda.synchronize()
e = time.time()
print('bone triton', (e-s)/100)
