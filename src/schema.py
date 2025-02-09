INPUT_SCHEMA = {
    "prompt": {
        "type": str,
        "required": True,
    },
    "token_repetition_penalty": {"type": float, "required": False, "default": 1.025},
    "token_repetition_range": {"type": int, "required": False, "default": -1},
    "token_repetition_decay": {"type": int, "required": False, "default": 0},
    "token_frequency_penalty": {"type": float, "required": False, "default": 0},
    "token_presence_penalty": {"type": float, "required": False, "default": 0},

    "temperature": {"type": float, "required": False, "default": 0.8},
    "smoothing_factor": {"type": float, "required": False, "default": 0},
    "min_temp": {"type": float, "required": False, "default": 0},
    "max_temp": {"type": float, "required": False, "default": 0},
    "temp_exponent": {"type": float, "required": False, "default": 1},
    
    "top_k": {"type": int, "required": False, "default": 50},
    "top_p": {"type": float, "required": False, "default": 0.8},
    "top_a": {"type": float, "required": False, "default": 0},
    "min_p": {"type": float, "required": False, "default": 0},
    
    "max_new_tokens": {"type": int, "required": False, "default": 1024},
    
}
