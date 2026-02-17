# Similar Image Finder & Cleanup Tool 🖼️

A Gradio-based application that uses AI vision models to identify similar images in a folder and helps you manage duplicates.

## Features

- 🔍 **Smart Detection**: Uses CLIP (vision-language model) to find similar images regardless of size, format, or brightness
- 📊 **Visual Review**: Browse groups of similar images in an intuitive interface
- ✅ **Selective Deletion**: Choose exactly which images to delete from each group
- ⚡ **Quick Cleanup**: One-click option to keep the largest image and delete the rest
- 🎯 **Adjustable Sensitivity**: Control how strict the similarity matching should be

## Installation

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt --break-system-packages
```

Or install packages individually:

```bash
pip install gradio Pillow numpy sentence-transformers torch --break-system-packages
```

### Step 2: Run the Application

```bash
python similar_image_finder.py
```

The app will open in your browser automatically at `http://localhost:7860`

## How to Use

### 1. Analyze Your Folder

- Enter the full path to your image folder in the text box
- Adjust the similarity threshold (0.85 is recommended):
  - **Lower values** (0.5-0.7): More lenient, finds images that are somewhat similar
  - **Higher values** (0.85-0.95): Stricter, only finds very similar images
- Click **"🔍 Analyze Folder"**

### 2. Review Similar Groups

- Use the slider to navigate between different groups of similar images
- Each group shows images that the AI detected as similar
- The gallery displays all images in the current group

### 3. Delete Images

**Option A: Manual Selection**
- Click on images in the gallery to select them for deletion (they'll be highlighted)
- Selected images will be listed in the "Selected for Deletion" box
- Click **"🗑️ Delete Selected Images"** to permanently remove them

**Option B: Keep Largest**
- Click **"✨ Keep Largest, Delete Others"** to automatically keep the largest file and delete all others in the group
- Useful when you have multiple copies at different resolutions

### 4. Safety Tips

⚠️ **Important**: Deleted images cannot be recovered! Consider:
- Making a backup of your folder first
- Testing on a small folder first
- Using a copy of your images rather than the originals

## How It Works

The application uses the CLIP (Contrastive Language-Image Pre-training) model to:

1. **Generate embeddings**: Convert each image into a numerical representation that captures its visual content
2. **Compare images**: Calculate similarity scores between all image pairs
3. **Group similar images**: Cluster images that exceed the similarity threshold
4. **Enable review**: Present groups in an easy-to-navigate interface

### Fallback Mode

If `sentence-transformers` is not installed, the app falls back to basic histogram-based comparison. For best results, ensure all dependencies are installed.

## Supported Image Formats

- JPEG/JPG
- PNG
- GIF
- BMP
- WebP
- TIFF

## Example Use Cases

- **Photographer's workflow**: Find duplicate shots from a photoshoot
- **Screenshot cleanup**: Remove redundant screenshots with minor differences
- **Download folder**: Clean up multiple downloads of the same image
- **Social media**: Find and remove duplicate memes or saved images
- **Design assets**: Consolidate multiple versions of logos or graphics

## Technical Details

- **Model**: CLIP ViT-B/32 (Vision Transformer)
- **Similarity metric**: Cosine similarity on normalized embeddings
- **Memory**: Processes images efficiently with batching
- **Privacy**: All processing happens locally on your machine

## Troubleshooting

**Problem**: "sentence-transformers not available"
- **Solution**: Install with `pip install sentence-transformers --break-system-packages`

**Problem**: CUDA/GPU errors
- **Solution**: The app works fine on CPU. If you want GPU acceleration, ensure PyTorch is properly installed with CUDA support

**Problem**: Out of memory
- **Solution**: Process fewer images at once or close other applications

**Problem**: No similar groups found
- **Solution**: Lower the similarity threshold or verify your images are actually similar

## License

Free to use and modify for personal and commercial purposes.
