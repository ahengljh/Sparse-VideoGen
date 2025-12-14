#!/usr/bin/env python3
"""
Quality evaluation for CTAA experiments.

Compares generated videos against baseline using:
- PSNR (Peak Signal-to-Noise Ratio) - pixel-level similarity
- SSIM (Structural Similarity Index) - structural similarity
- LPIPS (Learned Perceptual Image Patch Similarity) - perceptual similarity

Usage:
    python experiments/quality_eval.py --baseline path/to/baseline.mp4 --target path/to/ctaa.mp4
    python experiments/quality_eval.py --results_dir experiment_results/quality_evaluation_xxx
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# Conditional imports with fallbacks
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available.")

try:
    from skimage.metrics import structural_similarity as ssim_func
    from skimage.metrics import peak_signal_noise_ratio as psnr_func
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. Install with: pip install scikit-image")

try:
    import lpips
    HAS_LPIPS = True
except ImportError:
    HAS_LPIPS = False
    print("Warning: LPIPS not available. Install with: pip install lpips")


def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Load all frames from a video file."""
    if not HAS_CV2:
        raise ImportError("OpenCV required for video loading")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images."""
    if HAS_SKIMAGE:
        return psnr_func(img1, img2, data_range=255)
    else:
        # Manual implementation
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(255.0 / np.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM between two images."""
    if HAS_SKIMAGE:
        # For color images, compute SSIM per channel and average
        return ssim_func(img1, img2, channel_axis=2, data_range=255)
    else:
        raise ImportError("scikit-image required for SSIM computation")


class LPIPSEvaluator:
    """LPIPS perceptual similarity evaluator."""

    def __init__(self, net: str = 'alex', device: str = 'cuda'):
        if not HAS_LPIPS:
            raise ImportError("LPIPS library required. Install with: pip install lpips")

        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = lpips.LPIPS(net=net).to(self.device)
        self.model.eval()

    def compute(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Compute LPIPS distance between two images."""
        # Convert to tensor: [H, W, C] -> [1, C, H, W], range [-1, 1]
        def to_tensor(img):
            img = img.astype(np.float32) / 255.0 * 2 - 1  # [0,255] -> [-1,1]
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return img.to(self.device)

        t1 = to_tensor(img1)
        t2 = to_tensor(img2)

        with torch.no_grad():
            distance = self.model(t1, t2)

        return distance.item()


def evaluate_video_pair(
    baseline_path: str,
    target_path: str,
    use_lpips: bool = True,
    sample_rate: int = 1,  # Evaluate every N frames
) -> Dict:
    """
    Evaluate quality metrics between baseline and target videos.

    Args:
        baseline_path: Path to baseline (full attention) video
        target_path: Path to target (CTAA) video
        use_lpips: Whether to compute LPIPS (slower but more perceptually meaningful)
        sample_rate: Evaluate every N frames (1 = all frames)

    Returns:
        Dictionary containing per-frame and aggregate metrics
    """
    print(f"Loading baseline: {baseline_path}")
    baseline_frames = load_video_frames(baseline_path)
    print(f"Loading target: {target_path}")
    target_frames = load_video_frames(target_path)

    if len(baseline_frames) != len(target_frames):
        print(f"Warning: Frame count mismatch ({len(baseline_frames)} vs {len(target_frames)})")
        min_frames = min(len(baseline_frames), len(target_frames))
        baseline_frames = baseline_frames[:min_frames]
        target_frames = target_frames[:min_frames]

    num_frames = len(baseline_frames)
    print(f"Evaluating {num_frames} frames...")

    # Initialize LPIPS if needed
    lpips_evaluator = None
    if use_lpips and HAS_LPIPS and HAS_TORCH:
        try:
            lpips_evaluator = LPIPSEvaluator()
        except Exception as e:
            print(f"Warning: Could not initialize LPIPS: {e}")
            use_lpips = False

    # Compute per-frame metrics
    psnr_values = []
    ssim_values = []
    lpips_values = []

    for i in range(0, num_frames, sample_rate):
        baseline_frame = baseline_frames[i]
        target_frame = target_frames[i]

        # PSNR
        psnr = compute_psnr(baseline_frame, target_frame)
        psnr_values.append(psnr)

        # SSIM
        if HAS_SKIMAGE:
            ssim_val = compute_ssim(baseline_frame, target_frame)
            ssim_values.append(ssim_val)

        # LPIPS
        if lpips_evaluator:
            lpips_val = lpips_evaluator.compute(baseline_frame, target_frame)
            lpips_values.append(lpips_val)

        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"  Frame {i+1}/{num_frames}: PSNR={psnr:.2f}dB", end="")
            if ssim_values:
                print(f", SSIM={ssim_values[-1]:.4f}", end="")
            if lpips_values:
                print(f", LPIPS={lpips_values[-1]:.4f}", end="")
            print()

    # Aggregate metrics
    results = {
        'baseline_path': baseline_path,
        'target_path': target_path,
        'num_frames': num_frames,
        'sample_rate': sample_rate,
        'metrics': {
            'psnr': {
                'per_frame': psnr_values,
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values)),
            },
        }
    }

    if ssim_values:
        results['metrics']['ssim'] = {
            'per_frame': ssim_values,
            'mean': float(np.mean(ssim_values)),
            'std': float(np.std(ssim_values)),
            'min': float(np.min(ssim_values)),
            'max': float(np.max(ssim_values)),
        }

    if lpips_values:
        results['metrics']['lpips'] = {
            'per_frame': lpips_values,
            'mean': float(np.mean(lpips_values)),
            'std': float(np.std(lpips_values)),
            'min': float(np.min(lpips_values)),
            'max': float(np.max(lpips_values)),
        }

    return results


def evaluate_experiment_results(results_dir: str, output_file: str = None):
    """
    Evaluate all experiments in a results directory against baseline.

    Expects directory structure:
        results_dir/
            prompt_00_xxx/
                baseline.mp4
                ctaa_default.mp4
                ctaa_conservative.mp4
                ...
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_evaluations = []

    # Find all prompt directories
    prompt_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("prompt_")])

    for prompt_dir in prompt_dirs:
        print(f"\n{'='*60}")
        print(f"Evaluating: {prompt_dir.name}")
        print(f"{'='*60}")

        # Find baseline video
        baseline_path = prompt_dir / "baseline.mp4"
        if not baseline_path.exists():
            print(f"Warning: No baseline found in {prompt_dir}")
            continue

        # Find all other videos to compare
        target_videos = [f for f in prompt_dir.glob("*.mp4") if f.name != "baseline.mp4"]

        for target_path in target_videos:
            print(f"\nComparing: baseline vs {target_path.name}")

            try:
                eval_result = evaluate_video_pair(
                    str(baseline_path),
                    str(target_path),
                    use_lpips=True,
                    sample_rate=1,
                )
                eval_result['prompt_dir'] = prompt_dir.name
                eval_result['config_name'] = target_path.stem
                all_evaluations.append(eval_result)

                # Print summary
                print(f"\n  Summary for {target_path.name}:")
                print(f"    PSNR:  {eval_result['metrics']['psnr']['mean']:.2f} dB (std: {eval_result['metrics']['psnr']['std']:.2f})")
                if 'ssim' in eval_result['metrics']:
                    print(f"    SSIM:  {eval_result['metrics']['ssim']['mean']:.4f} (std: {eval_result['metrics']['ssim']['std']:.4f})")
                if 'lpips' in eval_result['metrics']:
                    print(f"    LPIPS: {eval_result['metrics']['lpips']['mean']:.4f} (std: {eval_result['metrics']['lpips']['std']:.4f})")

            except Exception as e:
                print(f"Error evaluating {target_path}: {e}")

    # Save all evaluations
    if output_file is None:
        output_file = results_dir / "quality_evaluation.json"

    with open(output_file, 'w') as f:
        json.dump({
            'results_dir': str(results_dir),
            'evaluations': all_evaluations,
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Quality evaluation saved to: {output_file}")
    print(f"{'='*60}")

    return all_evaluations


def print_quality_summary(evaluations: List[Dict]):
    """Print a summary table of quality metrics."""
    print("\n" + "=" * 80)
    print("QUALITY EVALUATION SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} {'PSNR (dB)':<15} {'SSIM':<15} {'LPIPS':<15}")
    print("-" * 80)

    # Group by config
    config_metrics = {}
    for eval_result in evaluations:
        config = eval_result.get('config_name', 'unknown')
        if config not in config_metrics:
            config_metrics[config] = {'psnr': [], 'ssim': [], 'lpips': []}

        config_metrics[config]['psnr'].append(eval_result['metrics']['psnr']['mean'])
        if 'ssim' in eval_result['metrics']:
            config_metrics[config]['ssim'].append(eval_result['metrics']['ssim']['mean'])
        if 'lpips' in eval_result['metrics']:
            config_metrics[config]['lpips'].append(eval_result['metrics']['lpips']['mean'])

    # Print aggregated results
    for config, metrics in sorted(config_metrics.items()):
        psnr_str = f"{np.mean(metrics['psnr']):.2f} +/- {np.std(metrics['psnr']):.2f}"
        ssim_str = f"{np.mean(metrics['ssim']):.4f}" if metrics['ssim'] else "N/A"
        lpips_str = f"{np.mean(metrics['lpips']):.4f}" if metrics['lpips'] else "N/A"
        print(f"{config:<25} {psnr_str:<15} {ssim_str:<15} {lpips_str:<15}")

    print("=" * 80)
    print("Note: Higher PSNR/SSIM = better quality, Lower LPIPS = better perceptual quality")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate video quality for CTAA experiments")
    parser.add_argument("--baseline", type=str, help="Path to baseline video")
    parser.add_argument("--target", type=str, help="Path to target video to compare")
    parser.add_argument("--results_dir", type=str, help="Directory containing experiment results")
    parser.add_argument("--output", type=str, help="Output JSON file for results")
    parser.add_argument("--no_lpips", action="store_true", help="Skip LPIPS computation")
    parser.add_argument("--sample_rate", type=int, default=1, help="Evaluate every N frames")

    args = parser.parse_args()

    if args.baseline and args.target:
        # Single video pair comparison
        results = evaluate_video_pair(
            args.baseline,
            args.target,
            use_lpips=not args.no_lpips,
            sample_rate=args.sample_rate,
        )

        print("\n" + "=" * 60)
        print("QUALITY METRICS SUMMARY")
        print("=" * 60)
        print(f"PSNR:  {results['metrics']['psnr']['mean']:.2f} dB")
        if 'ssim' in results['metrics']:
            print(f"SSIM:  {results['metrics']['ssim']['mean']:.4f}")
        if 'lpips' in results['metrics']:
            print(f"LPIPS: {results['metrics']['lpips']['mean']:.4f}")
        print("=" * 60)

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {args.output}")

    elif args.results_dir:
        # Batch evaluation of experiment results
        evaluations = evaluate_experiment_results(args.results_dir, args.output)
        print_quality_summary(evaluations)

    else:
        parser.print_help()
        print("\nExamples:")
        print("  python experiments/quality_eval.py --baseline baseline.mp4 --target ctaa.mp4")
        print("  python experiments/quality_eval.py --results_dir experiment_results/quality_xxx/")


if __name__ == "__main__":
    main()
