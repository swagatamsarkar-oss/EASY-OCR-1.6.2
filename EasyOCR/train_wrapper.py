"""
Fixed EasyOCR Training Wrapper for LMDB Format
This version properly handles LMDB datasets instead of CSV format

Upload to: /content/EasyOCR/train_wrapper.py
"""

import os

import os
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ['OMP_NUM_THREADS'] = '1'

import sys
import argparse

# Add the trainer directory to the path
sys.path.insert(0, '/content/EasyOCR/trainer')

from train import train

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', default='anpr_model', help='Where to store logs and models')
    parser.add_argument('--train_data', required=True, help='path to training dataset')
    parser.add_argument('--valid_data', required=True, help='path to validation dataset')
    parser.add_argument('--manualSeed', type=int, default=1111, help='for random seed setting')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--num_iter', type=int, default=300000, help='number of iterations to train for')
    parser.add_argument('--valInterval', type=int, default=2000, help='Interval between each validation')
    parser.add_argument('--saved_model', default='', help="path to model to continue training")
    parser.add_argument('--FT', action='store_true', help='whether to do fine-tuning')
    parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is Adadelta)')
    parser.add_argument('--optim', type=str, default='adadelta', help='optimizer: adam, adadelta, sgd')
    parser.add_argument('--lr', type=float, default=1, help='learning rate, default=1.0 for Adadelta')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
    parser.add_argument('--rho', type=float, default=0.95, help='decay rate rho for Adadelta. default=0.95')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps for Adadelta. default=1e-8')
    parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping value. default=5')
    parser.add_argument('--rgb', action='store_true', help='use rgb input')
    parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
    parser.add_argument('--Transformation', type=str, default='TPS', help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, default='ResNet', help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, default='BiLSTM', help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--decode', default='greedy', help='text decoding method: greedy or beamsearch')
    parser.add_argument('--Prediction', type=str, default='Attn', help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512, help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')
    
    # Image dimensions - REQUIRED
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
    parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
    
    # Data augmentation
    parser.add_argument('--contrast_adjust', type=float, default=0.0, help='contrast adjustment (0.0 = no adjustment)')
    
    # Data usage and selection
    parser.add_argument('--total_data_usage_ratio', type=float, default=1.0, help='ratio of total dataset to use')
    parser.add_argument('--select_data', type=str, default='/', help='select training data')
    parser.add_argument('--batch_ratio', type=str, default='1', help='batch ratio for each dataset')
    
    opt = parser.parse_args()
    if not hasattr(opt, 'decode'):
        opt.decode = 'greedy'
    
    # Remove trailing slashes
    opt.train_data = opt.train_data.rstrip('/')
    opt.valid_data = opt.valid_data.rstrip('/')
    
    # Handle adam flag - if --adam is set, use 'adam' as optimizer
    if opt.adam and opt.optim == 'adadelta':
        opt.optim = 'adam'
    
    # IMPORTANT: Keep select_data and batch_ratio as STRINGS
    # train.py will convert them to lists itself
    if not isinstance(opt.select_data, str):
        opt.select_data = str(opt.select_data)
    if not isinstance(opt.batch_ratio, str):
        opt.batch_ratio = str(opt.batch_ratio)
    
    return opt

if __name__ == '__main__':
    opt = get_args()
    
    print("="*80)
    print("ANPR CUSTOM MODEL TRAINING - LMDB FORMAT")
    print("="*80)
    print(f"Experiment: {opt.experiment_name}")
    print(f"Training data: {opt.train_data}")
    print(f"Validation data: {opt.valid_data}")
    print(f"Architecture: {opt.Transformation}-{opt.FeatureExtraction}-{opt.SequenceModeling}-{opt.Prediction}")
    print(f"Character set: {opt.character}")
    print(f"Batch size: {opt.batch_size}")
    print(f"Number of iterations: {opt.num_iter}")
    print("="*80)
    
    # Verify LMDB data exists
    import lmdb
    
    def verify_lmdb(path, name):
        """Verify LMDB dataset"""
        if not os.path.exists(path):
            print(f"✗ ERROR: {name} path does not exist: {path}")
            return False
        
        data_file = os.path.join(path, 'data.mdb')
        if not os.path.exists(data_file):
            print(f"✗ ERROR: data.mdb not found in {path}")
            print(f"   This should be an LMDB directory, not a parent directory")
            return False
        
        try:
            env = lmdb.open(path, readonly=True, lock=False, readahead=False, meminit=False)
            with env.begin() as txn:
                n_samples_key = 'num-samples'.encode()
                n_samples = txn.get(n_samples_key)
                
                if n_samples is None:
                    print(f"✗ ERROR: Invalid LMDB format in {path} - missing 'num-samples' key")
                    return False
                
                n_samples = int(n_samples.decode('utf-8'))
                print(f"✓ {name}: Valid LMDB with {n_samples} samples")
            env.close()
            return True
        except Exception as e:
            print(f"✗ ERROR reading {name} LMDB: {e}")
            return False
    
    # Check training data
    train_lmdb_path = opt.train_data.rstrip('/')
    if not verify_lmdb(train_lmdb_path, "Training data"):
        sys.exit(1)
    
    # Check validation data
    val_lmdb_path = opt.valid_data.rstrip('/')
    if not verify_lmdb(val_lmdb_path, "Validation data"):
        sys.exit(1)
    
    print("\n✓ All LMDB datasets verified successfully!")
    print("\nStarting training...\n")
    
    # Start training
    train(opt, amp=False)