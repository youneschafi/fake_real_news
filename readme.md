# fake_real_news
## Download Pretrained Weights

The `roberta-fake-vs-real-final/` folder is not included here (too large).  
Please download and extract the model from one of these locations:

- **Hugging Face Hub**:  
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="/path/to/local/model",
    repo_id="Chafiyounes/fakenews",
    repo_type="model",
)

git lfs install
git clone https://huggingface.co/Chafiyounes/fakenews


Or download the ZIP from https://huggingface.co/Chafiyounes/fakenews