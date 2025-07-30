from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, GenerationConfig
import os

model_name = "microsoft/phi-4"

print(f"Starting download/verification for {model_name}")

# Download tokenizer
print("\nChecking tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
print("✓ Tokenizer ready")

# Download configuration files
print("\nChecking configuration files...")
config = AutoConfig.from_pretrained(model_name)
generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)
print("✓ Configuration files ready")

# Download model files
print("\nChecking model files...")
model_info = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    local_files_only=False,  # Will download if missing
    offload_folder="offload",
    offload_state_dict=True,  # Don't load weights into memory
    config=config,
)
print("✓ Model files ready")

# Save configs locally if needed
config.save_pretrained("./model_cache")
generation_config.save_pretrained("./model_cache")

print("\nInstallation complete! Files are cached and ready to use.")

# Print some useful model info
print("\nModel details:")
print(f"Context length: {config.max_position_embeddings}")
print(f"Vocabulary size: {config.vocab_size}")
print(f"Model parameters: {model_info.num_parameters()}")