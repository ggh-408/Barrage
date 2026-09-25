"""Measure exact row-parallel RGB kernels on recorded images, process-local."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.screen_exact_latency_candidates import paired
from barrage_rl.rgb_capture_kernel import copy_rgb
from barrage_rl.foreground_kernel import fused_foreground,njit
from numba import prange,set_num_threads

@njit(cache=False,fastmath=False,parallel=True)
def copy_parallel(image):
    result=np.empty((image.shape[0],image.shape[1],3),np.uint8)
    for y in prange(image.shape[0]):
        for x in range(image.shape[1]):
            for c in range(3):result[y,x,c]=image[y,x,c]
    return result

@njit(cache=False,fastmath=False,parallel=True)
def foreground_parallel(image,lower,upper):
    result=np.empty(image.shape[:2],np.bool_)
    for y in prange(image.shape[0]):
        for x in range(image.shape[1]):
            hit=False
            for c in range(3):
                v=image[y,x,c]
                if v<=lower[c] or v>=upper[c]:hit=True
            result[y,x]=hit
    return result

if __name__=='__main__':
    import pygame
    with (ROOT/'diagnostics/rgb_blocks_20260925_112807_783291/rgb_samples.pkl').open('rb') as f:samples=pickle.load(f)
    capture=[];foreground=[];surfaces=[]
    for sample in samples:
        im=sample['image'];h,w=im.shape[:2]
        surface=pygame.Surface((w,h),depth=32);pygame.surfarray.blit_array(surface,im.transpose(1,0,2));surfaces.append(surface)
        capture.append((np.asarray(surface.get_view('3')).transpose(1,0,2),))
        bg=np.median(im[::max(1,h//32),::max(1,w//32),:3].reshape(-1,3),axis=0).astype(np.int16)
        foreground.append((im,np.floor(bg-28).astype(np.int64),np.ceil(bg+28).astype(np.int64)))
    results=[]
    for workers in (1,2,4,9):
        set_num_threads(workers)
        row=dict(workers=workers,capture=paired((copy_rgb,copy_parallel),capture,6),
                 foreground=paired((fused_foreground,foreground_parallel),foreground,6))
        results.append(row);print(json.dumps(row),flush=True)
    (ROOT/'diagnostics/exact_latency_screen_20260925/parallel_rgb.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
