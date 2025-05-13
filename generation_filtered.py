import os
import shutil
import glob
import argparse
from utils import process_variations
from mesh_utils import process_image  # make sure this module defines your process_image() as above

# Map your class indices to race folder names
CLASS_TO_RACE = {
    0: 'ST1',
    1: 'ST2',
    2: 'ST3',
    3: 'ST4',
    4: 'ST5',
    5: 'ST6',
    6: 'ST7',
    7: 'ST8',
}


def within_threshold(meas, target, rel_thresh):
    """
    Check if meas is within rel_thresh (fraction) of target.
    Uses absolute difference <= abs(target) * rel_thresh.
    If target is zero, falls back to abs(meas) <= rel_thresh.
    """
    if target == 0:
        return abs(meas) <= rel_thresh
    return abs(meas - target) <= abs(target) * rel_thresh


def main():
    parser = argparse.ArgumentParser(
        description="Generate images and keep only those within roll/pitch/yaw thresholds"
    )
    parser.add_argument('--root', default=os.path.join(os.getcwd(), 'gen_test'),
                        help='Root directory containing race subfolders (e.g. ST1, ST2, ...)')
    parser.add_argument('--class_index', type=int, default=2,
                        help='Class index to generate (key into CLASS_TO_RACE)')
    parser.add_argument('--start_id', type=int, default=0,
                        help='Start identity index')
    parser.add_argument('--end_id', type=int, default=10000,
                        help='End identity index (exclusive)')
    parser.add_argument('--roll', type=float, default=-162.095,
                        help='Target roll angle (degrees)')
    parser.add_argument('--pitch', type=float, default=-5.499,
                        help='Target pitch angle (degrees)')
    parser.add_argument('--yaw', type=float, default=1.125,
                        help='Target yaw angle (degrees)')
    parser.add_argument('--threshold', type=float, default=0.05,
                        help='Relative threshold fraction (e.g. 0.1 for ±10%%)')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of valid images to keep')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='How many identities to generate per iteration')
    args = parser.parse_args()

    # Validate and locate race folder
    if args.class_index not in CLASS_TO_RACE:
        raise ValueError(f"Invalid class_index {args.class_index}. Valid keys: {list(CLASS_TO_RACE.keys())}")
    race_folder = CLASS_TO_RACE[args.class_index]
    output_base = os.path.join(args.root, race_folder)

    # Variations settings
    variations_dict = {
        'eyes':  {'var_name': 'eyes',  'include': False, 'save_w': False},
        'mouth': {'var_name': 'mouth', 'include': False, 'save_w': False},
        'smile': {'var_name': 'smile', 'include': False, 'save_w': False},
        'pose':  {'var_name': 'pose',  'include': False, 'save_w': False},
    }

    accepted = 0
    current_id = args.start_id

    while accepted < args.count:
        batch_end = min(current_id + args.batch_size, args.end_id)
        print(f"Generating IDs {current_id} to {batch_end} in {race_folder}...")

        # Generate into the race subfolder
        process_variations(
            variations_dict,
            args.class_index,
            current_id,
            batch_end,
            output_dir=output_base
        )

        # Scan generated seed subfolders: cond{idx}_seed*
        for idx in range(current_id, batch_end):
            if accepted >= args.count:
                break
            pattern = os.path.join(output_base, f"cond{idx}_seed*")
            for subdir in glob.glob(pattern):
                img_path = os.path.join(subdir, 'original.png')
                print(f"Checking {img_path}...")
                if not os.path.isfile(img_path):
                    print(f"[WARN] missing image for ID {idx} at {subdir}")
                    continue

                metrics = process_image(img_path)
                if metrics is None:
                    print(f"[DISC] no face detected in {subdir}")
                    shutil.rmtree(subdir, ignore_errors=True)
                    continue

                ok_roll  = within_threshold(metrics['roll'],  args.roll,  args.threshold)
                ok_pitch = within_threshold(metrics['pitch'], args.pitch, args.threshold)
                ok_yaw   = within_threshold(metrics['yaw'],   args.yaw,   args.threshold)

                if ok_roll and ok_pitch and ok_yaw:
                    accepted += 1
                    print(f"[KEEP] {subdir} → roll={metrics['roll']:.1f}, "
                          f"pitch={metrics['pitch']:.1f}, yaw={metrics['yaw']:.1f} "
                          f"({accepted}/{args.count})")
                    if accepted >= args.count:
                        break
                else:
                    print(f"[DROP] {subdir} → roll={metrics['roll']:.1f}, "
                          f"pitch={metrics['pitch']:.1f}, yaw={metrics['yaw']:.1f}")
                    shutil.rmtree(subdir, ignore_errors=True)

        # Advance or wrap around
        current_id = batch_end if batch_end < args.end_id else args.start_id

    print(f"Done! Collected {accepted} valid images in {race_folder}.")

if __name__ == '__main__':
    main()