
from torch.utils.data import Dataset, DataLoader
from utils import parse_arguments, get_config
import subprocess
import os 
from huggingface_hub import hf_hub_download

# Download necessary models 
models = [
    './checkpoints/facenet_vggface2.pt',
    './checkpoints/shape_predictor_68_face_landmarks.dat',
    './checkpoints/network_conditional_skin_tones.pkl'
]

for model in models: 
    hf_hub_download(repo_id="pravsels/synthpar", 
                    filename=model.split('/')[-1], 
                    repo_type="model", 
                    local_dir=os.path.dirname(model))


command_line_args = parse_arguments()
config = get_config(command_line_args.config)

no_of_classes = 8

class_index = config.class_index
no_of_identities_start = config['no_of_identities'][0]
no_of_identities_end = config['no_of_identities'][1]

variations_dict = {
    'eyes':     {'var_name':  'eyes', 'include': True, 'save_w': False},
    'mouth':    {'var_name': 'mouth', 'include': True, 'save_w': False},
    'smile':    {'var_name': 'smile', 'include': True, 'save_w': False},
    'pose':     {'var_name':  'pose', 'include': True, 'save_w': False},
}

batch_size = config.batch_size

total_identities_per_class = no_of_identities_end - no_of_identities_start
identities_per_batch = total_identities_per_class // batch_size
remainder = total_identities_per_class % batch_size

print('Total identities:', total_identities_per_class)
print('Identities per batch:', identities_per_batch)
print('Remainder:', remainder)

processes = []
for batch_idx in range(batch_size):
    start_idx = no_of_identities_start + batch_idx * identities_per_batch + min(batch_idx, remainder)
    end_idx = start_idx + identities_per_batch + (1 if batch_idx < remainder else 0)

    print('start idx : ', start_idx)
    print('end idx : ', end_idx)
    
    process = subprocess.Popen(['python', '-c', f"from utils import process_variations; process_variations({variations_dict}, {class_index}, {start_idx}, {end_idx})"])
    processes.append(process)

for process in processes:
    process.wait()

