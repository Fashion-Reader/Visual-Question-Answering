"""
"""
from models.vqa_model import VQAModel

def get_model(model_str: str) -> 'model':
    
    if model_str == 'model':
        return Model
    elif model_str == 'vqa_model':
        return VQAModel


if __name__ == '__main__':
    pass

