import torch
import torch.nn as nn

class DALLE(nn.Module):
    def __init__(self, vocab_size, image_size, num_layers=2, hidden_size=256, num_heads=4):
        super(DALLE, self).__init__()
        
        # Text Encoder
        self.text_embedding = nn.Embedding(vocab_size, hidden_size)
        
        # Image Decoder
        self.image_decoder = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        
        self.fc = nn.Linear(hidden_size, image_size * image_size * 3)
        
    def forward(self, text_inputs):
        text_embedded = self.text_embedding(text_inputs)
        
        # Image Generation
        image_output = self.image_decoder(text_embedded, text_embedded)
        image_output = self.fc(image_output)
        
        return image_output

# Example usage
vocab_size = 10000  # Size of the vocabulary
image_size = 64    # Size of the generated image (64x64)

model = DALLE(vocab_size, image_size)

# Create some dummy input
batch_size = 16
seq_length = 20
dummy_text_input = torch.randint(0, vocab_size, (batch_size, seq_length))

# Forward pass
generated_images = model(dummy_text_input)
print("Generated Images Shape:", generated_images.shape)
