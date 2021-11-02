import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.allow_tf32 = False
    data = torch.randn([1, 256, 128, 128], dtype=torch.float, device='cuda', requires_grad=True)
    net = torch.nn.Conv2d(256, 256, kernel_size=[3, 3], padding=[1, 1], stride=[1, 1], dilation=[1, 1], groups=1)
    net = net.cuda().float()
    out = net(data)
    out.backward(torch.randn_like(out))
    torch.cuda.synchronize()
                         
if __name__ == "__main__":
    main()