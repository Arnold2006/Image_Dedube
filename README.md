# Similar Image Finder & Cleanup Tool 🖼️

A desktop Python application with a graphical interface (tkinter) that uses AI vision models to identify similar images in a folder and helps you manage duplicates.

## Features

- 🔍 **Smart Detection**: Uses CLIP (vision-language model) to find similar images regardless of size, format, or brightness
- 🖥️ **Native Desktop App**: Clean GUI using Python's built-in tkinter (no browser needed)
- 📊 **Visual Review**: Browse groups of similar images with thumbnail previews
- ✅ **Selective Deletion**: Click images to select them for deletion
- ⚡ **Quick Cleanup**: One-click option to keep the largest image and delete the rest
- 🎯 **Adjustable Sensitivity**: Control how strict the similarity matching should be

## Installation

### Requirements
- Python 3.8 or higher
- tkinter (usually included with Python)

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install Pillow numpy sentence-transformers torch
```

**Note:** If you don't have `sentence-transformers` installed, the app will use a basic fallback method (less accurate but functional).

### Run the Application

```bash
python app.py
```

A window will open with the application interface.

## How to Use

### 1. Select Your Folder

1. Click **"Browse..."** to select the folder containing your images
2. Adjust the **similarity threshold** slider if desired:
   - **Lower values** (0.5-0.7): More lenient, finds images that are somewhat similar
   - **Higher values** (0.85-0.95): Stricter, only finds very similar images
3. Click **"🔍 Analyze Folder"**

The app will analyze all images in the folder and its subfolders.

### 2. Review Similar Groups

- The app will display the first group of similar images
- Use **"← Previous"** and **"Next →"** buttons to navigate between groups
- Each image shows:
  - Thumbnail preview
  - Filename
  - File size in KB

### 3. Select Images to Delete

**Click on any image** to select it for deletion:
- Selected images will have a thick border
- You can select multiple images
- Click again to deselect

The status bar shows how many images are selected.

### 4. Delete Images

**Option A: Delete Selected**
- Select the images you want to remove by clicking them
- Click **"🗑️ Delete Selected Images"**
- Confirm the deletion

**Option B: Keep Largest**
- Click **"✨ Keep Largest, Delete Others"**
- The app will automatically keep the image with the largest file size and delete all others in the group
- Useful when you have the same image at different resolutions

**Option C: Clear Selection**
- Click **"Clear Selection"** to deselect all images and start over

### 5. Navigate Between Groups

- After processing a group, use the navigation buttons to move to the next group
- Groups with only 1 image remaining are automatically removed

## Safety Tips

⚠️ **Important**: Deleted images **cannot be recovered**! Consider:
- Making a backup of your folder first
- Testing on a small folder first
- Using a copy of your images rather than the originals

## How It Works

The application uses the CLIP (Contrastive Language-Image Pre-training) model to:

1. **Generate embeddings**: Convert each image into a numerical representation that captures its visual content
2. **Compare images**: Calculate similarity scores between all image pairs using cosine similarity
3. **Group similar images**: Cluster images that exceed the similarity threshold
4. **Enable review**: Present groups in an easy-to-navigate desktop interface

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

- **GUI Framework**: tkinter (native Python, cross-platform)
- **Model**: CLIP ViT-B/32 (Vision Transformer)
- **Similarity metric**: Cosine similarity on normalized embeddings
- **Memory**: Processes images efficiently with progress feedback
- **Privacy**: All processing happens locally on your machine

## Troubleshooting

**Problem**: "sentence-transformers not available"
- **Solution**: Install with `pip install sentence-transformers`
- The app will work with basic comparison, but CLIP gives better results

**Problem**: CUDA/GPU errors with PyTorch
- **Solution**: The app works fine on CPU. If you want GPU acceleration, ensure PyTorch is properly installed with CUDA support

**Problem**: Out of memory
- **Solution**: Process folders with fewer images at once, or close other applications

**Problem**: No similar groups found
- **Solution**: Lower the similarity threshold or verify your images are actually similar

**Problem**: tkinter not found
- **Solution**: 
  - **Windows/Mac**: tkinter comes with Python
  - **Linux**: Install with `sudo apt-get install python3-tk` (Ubuntu/Debian) or equivalent

## Screenshots

The app features:
- Clean, intuitive interface
- Thumbnail grid view of similar images
- Easy click-to-select interaction
- Progress bar during analysis
- Status updates throughout the process

## License

Free to use and modify for personal and commercial purposes.
