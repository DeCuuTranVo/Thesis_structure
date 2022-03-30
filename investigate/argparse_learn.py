import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--kernel-type', type=str, default='9c_b4ns_448_ext_15ep-newfold')
    parser.add_argument('--data-dir', type=str, default="./data/")
    parser.add_argument('--data-folder', type=int, default=512)
    parser.add_argument('--image-size', type=int, default=448)
    parser.add_argument('--enet-type', type=str, default='tf_efficientnet_b3')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--init-lr', type=float, default=3e-5)
    parser.add_argument('--out-dim', type=int, default=9)
    parser.add_argument('--n-epochs', type=int, default=15)
    parser.add_argument('--use-amp', action='store_true')
    parser.add_argument('--use-meta', action='store_true')
    parser.add_argument('--DEBUG', action='store_true')
    parser.add_argument('--model-dir', type=str, default='./weights')
    parser.add_argument('--log-dir', type=str, default='./logs')
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, default='0')
    parser.add_argument('--fold', type=str, default='0,1,2,3,4')
    parser.add_argument('--n-meta-dim', type=str, default='512,128')

    args, _ = parser.parse_known_args()
    return args