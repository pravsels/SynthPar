import os
import shutil
import glob
import argparse
from utils import process_variations
from mesh_utils import process_image  # make sure this module defines your process_image() as above

# Map your class indices to race folder names
CLASS_TO_ST = {
    0: 'ST1',
    1: 'ST2',
    2: 'ST3',
    3: 'ST4',
    4: 'ST5',
    5: 'ST6',
    6: 'ST7',
    7: 'ST8',
}


def within_threshold(measured, target, abs_thresh_deg):
    return abs(measured - target) <= abs_thresh_deg

def main():
    parser = argparse.ArgumentParser(
        description="Generate images and keep only those within roll/pitch/yaw thresholds"
    )
    parser.add_argument('--root', default=os.path.join(os.getcwd(), 'gen_test'),
                        help='Root directory containing race subfolders (e.g. ST1, ST2, ...)')
    parser.add_argument('--class_index', type=int, default=2,
                        help='Class index to generate (key into CLASS_TO_ST)')
    parser.add_argument('--start_id', type=int, default=17,
                        help='Start identity index')
    parser.add_argument('--end_id', type=int, default=100000,
                        help='End identity index (exclusive)')
    parser.add_argument('--roll', type=float, default=-162.095,
                        help='Target roll angle (degrees)')
    parser.add_argument('--pitch', type=float, default=-5.499,
                        help='Target pitch angle (degrees)')
    parser.add_argument('--yaw', type=float, default=1.125,
                        help='Target yaw angle (degrees)')
    parser.add_argument('--threshold', type=float, default=10,
                        help='Degrees treshold')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of valid images to keep')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='How many identities to generate per iteration')
    args = parser.parse_args()

    # Validate and locate race folder
    if args.class_index not in CLASS_TO_ST:
        raise ValueError(f"Invalid class_index {args.class_index}. Valid keys: {list(CLASS_TO_ST.keys())}")
    st_folder = CLASS_TO_ST[args.class_index]
    output_base = os.path.join(args.root, st_folder)

    # Variations settings
    variations_dict = {
        'eyes':  {'var_name': 'eyes',  'include': False, 'save_w': False},
        'mouth': {'var_name': 'mouth', 'include': False, 'save_w': False},
        'smile': {'var_name': 'smile', 'include': False, 'save_w': False},
        'pose':  {'var_name': 'pose',  'include': False, 'save_w': False},
    }

    accepted = 0
    next_id = args.start_id

    while accepted < args.count:

        pattern = os.path.join(output_base, f"cond{args.class_index}_seed{next_id}")
        for old in glob.glob(pattern):
            shutil.rmtree(old, ignore_errors=True)

        print(f"Generating sample for cond{next_id}...")
        process_variations(
            variations_dict,
            args.class_index,
            next_id,
            next_id + 1,
            output_dir=args.root
        )
        # Check for new seed output
        seeds = glob.glob(pattern)
        # import pdb; pdb.set_trace()
        if not seeds:
            print(f"[WARN] no output for cond{next_id}")
            break 
        # Pick latest by modification time
        seeds.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        subdir = seeds[0]
        img_path = os.path.join(subdir, 'original.png')
        print(f"Checking {os.path.basename(subdir)}...")
        if not os.path.isfile(img_path):
            print(f"[WARN] missing original.png in {subdir}")
            break 

        metrics = process_image(img_path)
        print(metrics)
        if metrics is None:
            print(f"[DROP] no face detected in {os.path.basename(subdir)}")
            shutil.rmtree(subdir, ignore_errors=True)
        else:
            ok = (within_threshold(metrics['roll'],  args.roll,  args.threshold) and
                  within_threshold(metrics['pitch'], args.pitch, args.threshold) and
                  within_threshold(metrics['yaw'],   args.yaw,   args.threshold))
            if ok:
                accepted += 1
                print(f"[KEEP] {os.path.basename(subdir)} "
                      f"roll={metrics['roll']:.1f}, pitch={metrics['pitch']:.1f}, yaw={metrics['yaw']:.1f} "
                      f"({accepted}/{args.count})")
            else:
                print(f"[DROP] {os.path.basename(subdir)} "
                      f"roll={metrics['roll']:.1f}, pitch={metrics['pitch']:.1f}, yaw={metrics['yaw']:.1f}")
                shutil.rmtree(subdir, ignore_errors=True)

        next_id += 1

    print(f"Done! Collected {accepted} valid images in {st_folder}.")

if __name__ == '__main__':
    main()
