#!/bin/bash
# Push RAG Chatbot Backend to Hugging Face Spaces

# Install Hugging Face CLI if not installed
if ! command -v huggingface &> /dev/null; then
    echo "Installing Hugging Face CLI..."
    pip install -U huggingface_hub[cli]
fi

# Login to Hugging Face
echo "Please login to Hugging Face:"
huggingface-cli login

# Create Space and push
echo "Creating/updating Hugging Face Space..."
huggingface-cli space create rag-chatbot-backend \
    --sdk docker \
    --title "RAG Chatbot Backend" \
    --emoji "🤖"

# Clone the space repo and push
echo "Pushing code to Hugging Face Spaces..."
huggingface-cli space upload rag-chatbot-backend ./rag-chatbot-backend/ \
    --repo-type space \
    --repo-id "YOUR_HF_USERNAME/rag-chatbot-backend"

echo ""
echo "Done! Your Space should be available at:"
echo "https://huggingface.co/spaces/YOUR_HF_USERNAME/rag-chatbot-backend"
